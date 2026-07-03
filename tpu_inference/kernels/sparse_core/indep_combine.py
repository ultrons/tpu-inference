# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sparse MoE combine via the `indep` (no cross-tile carry) SparseCore kernel.

The production combine (`ragged_gather_reduce`) carries a cross-tile serial
recurrence (`prev_iter_last_row` + previous-dst) so that a reduce-group may
straddle a tile boundary. That recurrence blocks pipelining tile N+1's gather
under tile N's reduce. The `indep` kernel drops the carry (row 0 always starts a
fresh run), which is only correct when reduce-groups never straddle a tile
boundary. For SPARSE (EP>1) routing that requires a preprocess:
`prep_div_dummy` divisor-pads each token's run to {1,2,4,8,16} and dummy-fills so
every length-group is a multiple of the SC tile (16), realized with one
argsort->gather. `run_prepacked` then runs the indep kernel on that layout.

Wired behind `envs.MOE_COMBINE_INDEP` at the fused_moe_gmm sparse combine site.
Ported from perf-drills/gather/combine/{prep_dummy,prepacked_runner}.py.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import ragged_gather_reduce_indep as R

_TILE = 16  # SC row-tile granularity the divisor-pad aligns each length-group to.


@functools.partial(jax.jit, static_argnames=("K", "RP"))
def prep_div_dummy(indices, weights, valid, K, RP):
    """Divisor-pad + dummy-fill so `indep` is correct on sparse routing.

    Buckets each token's valid-count V into a padded run length L in
    {1,2,4,8,16}, aligns each per-length group to a multiple of _TILE with
    weight-0 dummy rows, and realizes the tile-aligned layout with one
    argsort->gather. Returns (src_indices, dst_indices, weights, nrows) for
    `run_prepacked`. `indices.shape[0]` must be a multiple of RP*K (the caller
    `indep_combine` pads to guarantee this).
    """
    P = indices.shape[0]
    rp = P // RP
    nt = rp // K
    TILE = _TILE
    v3 = valid.reshape(RP, nt, K).astype(jnp.int32)
    V = v3.sum(-1)
    L = jnp.where(V <= 1, 1,
                  jnp.where(V <= 2, 2, jnp.where(V <= 4, 4,
                                                 jnp.where(V <= 8, 8, 16))))
    Lz = jnp.where(V > 0, L, 0)
    rankb = jnp.broadcast_to(jnp.arange(K), (RP, nt, K))
    keptb = (rankb < Lz[:, :, None])
    # per-length-group counts / aligned starts
    ntok = jnp.stack([(Lz == lv).sum(-1) for lv in (1, 2, 4, 8)], -1)  # (RP,4)
    cnt = ntok * jnp.array([1, 2, 4, 8])
    pad = (TILE - cnt % TILE) % TILE
    aligned = cnt + pad
    gbase = jnp.cumsum(aligned, -1) - aligned
    cumpad = jnp.cumsum(pad, -1)
    cumpad_excl = cumpad - pad
    total_pad = cumpad[..., -1:]
    # token rank within its length group + group base per token
    trank = jnp.zeros((RP, nt), jnp.int32)
    base_tok = jnp.zeros((RP, nt), jnp.int32)
    for i, lv in enumerate((1, 2, 4, 8)):
        m = (Lz == lv).astype(jnp.int32)
        trank = trank + jnp.where(Lz == lv, jnp.cumsum(m, -1) - 1, 0)
        base_tok = base_tok + jnp.where(Lz == lv, gbase[:, i:i + 1], 0)
    kept_key = (base_tok[:, :, None] + (trank * Lz)[:, :, None] + rankb).reshape(
        RP, rp)  # kept -> run position
    # excess invalids -> dummies fill each group's padding region
    excess = (~keptb).reshape(RP, rp)
    exrank = jnp.cumsum(excess.astype(jnp.int32), -1) - 1
    dg = jnp.clip((cumpad[:, None, :] <= exrank[:, :, None]).sum(-1), 0, 3)
    dummy_key = jnp.zeros((RP, rp), jnp.int32)  # sum-of-where: no computed gather
    for g in range(4):
        dummy_key = dummy_key + jnp.where(
            dg == g, gbase[:, g:g + 1] + cnt[:, g:g + 1] +
            (exrank - cumpad_excl[:, g:g + 1]), 0)
    excess_key = jnp.where(excess & (exrank < total_pad), dummy_key, rp)
    key = jnp.where(keptb.reshape(RP, rp), kept_key, excess_key)
    # dst per slot: kept -> token, dummy -> per-partition sink
    tokb = jnp.broadcast_to(jnp.arange(nt)[None, :, None],
                            (RP, nt, K)).reshape(RP, rp)
    dst_slot = jnp.where(
        keptb.reshape(RP, rp), tokb + jnp.arange(RP)[:, None] * nt,
        RP * nt + jnp.arange(RP)[:, None])
    # ONE argsort -> gather (the fast path)
    order = jnp.argsort(key, axis=-1)
    og = (order + jnp.arange(RP)[:, None] * rp).reshape(-1)
    src = indices[og]
    w = weights[og].astype(jnp.float32)
    dst = jnp.take_along_axis(dst_slot, order, axis=-1).reshape(-1)
    nrows = jnp.pad(aligned.sum(-1).astype(jnp.int32), (0, 16 - RP))
    return src, dst, w, nrows


@functools.partial(jax.jit, static_argnames=("reduce_group_size", "num_out_rows"))
def run_prepacked(x, src_indices, dst_indices, topk_weights,
                  num_rows_per_partition, mask, reduce_group_size, num_out_rows):
    """Run the `indep` kernel on an already tile-aligned (prepacked) layout."""
    sc_info = pltpu.get_tpu_info().sparse_core
    hidden_size = x.shape[-1]
    input_size = x.shape[0]  # BEFORE padding (line below)
    padded_input_size = src_indices.shape[0]
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = 8
    num_rows_partitions = num_cores // num_column_partitions
    aligned_hidden_size = R._align_to(hidden_size, 128 * num_column_partitions)
    col_size = aligned_hidden_size // num_column_partitions
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    x = jnp.pad(x, ((0, padded_input_size - x.shape[0]),
                    (0, aligned_hidden_size - hidden_size)),
                constant_values=0)
    vm = plsc.VectorSubcoreMesh(num_cores=sc_info.num_cores,
                                num_subcores=sc_info.num_subcores,
                                core_axis_name="core",
                                subcore_axis_name="subcore")
    out = pl.kernel(
        functools.partial(R.main_kernel,
                          core_axis_name=vm.core_axis_name,
                          subcore_axis_name=vm.subcore_axis_name,
                          num_row_partitions=num_rows_partitions,
                          num_column_partitions=num_column_partitions),
        compiler_params=pltpu.CompilerParams(**R._COMPILER_PARAMS),
        cost_estimate=R.get_cost_estimate(
            padded_input_size=padded_input_size,
            aligned_hidden_size=aligned_hidden_size,
            reduce_group_size=reduce_group_size,
            input_dtype_bytes=dtype_bytes),
        mesh=vm,
        name="sc_ragged_gather_reduce_indep",
        **{
            R._OUT_KW:
                jax.ShapeDtypeStruct(
                    (max(num_out_rows, padded_input_size // reduce_group_size),
                     aligned_hidden_size), jnp.float32),
            R._SCRATCH_KW:
                dict(
                    num_rows_per_row_partition_vmem_ref=pltpu.VMEM(
                        (num_simd_lanes,), jnp.int32),
                    out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size),
                                            jnp.uint32),
                    prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size),
                                                           jnp.uint32),
                    src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,),
                                                    jnp.int32),
                    dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,),
                                                    jnp.int32),
                    topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,),
                                                     jnp.float32),
                    sem_ref=pltpu.SemaphoreType.DMA((2,))),
        },
    )(num_rows_per_partition, x, src_indices, dst_indices, topk_weights)
    out = out[:input_size // reduce_group_size, :hidden_size]
    return jnp.where(mask[:input_size // reduce_group_size, None],
                     out.astype(x.dtype), jnp.zeros_like(out, dtype=x.dtype))


def indep_combine(x, indices, topk_weights, valid_rows_mask, reduce_group_size):
    """Drop-in for `ragged_gather_reduce` using the `indep` kernel + preprocess.

    Same signature/return as `ragged_gather_reduce`. Pads the flat input up to a
    multiple of RP*K so `prep_div_dummy`'s (RP, tokens, K) reshape is valid for
    arbitrary chunk sizes, then slices the output back to the true token count.
    Falls back to the reference impl when SparseCore is unavailable.
    """
    K = reduce_group_size
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return R._fallback_implementation(x, indices, topk_weights,
                                          valid_rows_mask, reduce_group_size)
    num_cores = sc_info.num_cores * sc_info.num_subcores
    RP = num_cores // 8

    P = indices.shape[0]
    T = P // K  # true number of output (token) rows
    pad_mult = RP * K
    P_pad = ((P + pad_mult - 1) // pad_mult) * pad_mult
    if P_pad != P:
        n = P_pad - P
        indices = jnp.pad(indices, (0, n))
        topk_weights = jnp.pad(topk_weights, (0, n))
        valid_rows_mask = jnp.pad(valid_rows_mask, (0, n),
                                  constant_values=False)
    T_pad = P_pad // K

    src, dst, w, nrows = prep_div_dummy(indices, topk_weights, valid_rows_mask,
                                        K, RP)
    # Token has >=1 valid slot -> keep; all-invalid token -> zeroed output row.
    mask = jnp.any(valid_rows_mask.reshape(T_pad, K), axis=-1)
    num_out_rows = ((T_pad + RP + 7) // 8) * 8
    out = run_prepacked(x, src, dst, w, nrows, mask, K, num_out_rows)
    return out[:T]

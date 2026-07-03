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
"""Standalone correctness + timing A/B for the `indep` MoE combine.

Run on a v7x pod (SparseCore required):
    python tests/kernels/bench_indep_combine.py

Compares, at DeepSeek-R1 combine shapes (hidden=7168, topk=8, EP=8 sparse):
  * carry  = production ragged_gather_reduce_v2 (serial cross-tile carry)
  * indep  = indep_combine (divisor-pad preprocess + no-carry kernel)  [the candidate]
  * naive  = the indep kernel WITHOUT the preprocess (expected wrong on sparse)
against a numpy reference, then times carry-E2E vs indep-E2E (prep included).

CRITICAL: tests BOTH `valid-first` (each token's valid slots packed first) and
`scattered` (valid slots at arbitrary ranks) validity. prep_div_dummy keeps the
first-L ranks per token, so it is only correct if validity is valid-first. Real
combine masks (which of a token's topk experts are on this shard) are generally
scattered — if the `scattered` row below is large for indep, indep_combine needs
a per-token valid-first pre-sort before it can be wired for real.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as carry
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_indep import \
    ragged_gather_reduce as naive_indep
from tpu_inference.kernels.sparse_core.indep_combine import indep_combine

H, K, EP = 7168, 8, 8


def build(T, seed, pack):
    """Return (idx, wts, valid, xnp). pack='first' or 'scattered'."""
    INPUT = T * K
    rng = np.random.default_rng(seed)
    if pack == "first":
        counts = rng.binomial(K, 1.0 / EP, size=T)
        valid = (np.arange(K)[None, :] < counts[:, None]).reshape(-1)
    else:  # scattered — independent per-slot (matches real per-expert masking)
        valid = (rng.random((T, K)) < 1.0 / EP).reshape(-1)
    idx = rng.permutation(INPUT).astype(np.int32)
    wts = rng.random(INPUT).astype(np.float32)
    idx[~valid] = 0
    wts[~valid] = 0.0
    xnp = rng.standard_normal((INPUT, H)).astype(np.float32)
    return idx, wts, valid.astype(bool), xnp


def ref_np(idx, wts, valid, xnp, T):
    contrib = xnp[idx] * (wts * valid)[:, None]
    return contrib.reshape(T, K, H).sum(1)


def relerr(o, ref):
    return float(np.abs(o - ref).max() / (np.abs(ref).max() + 1e-9))


def wall(fn, it=20):
    for _ in range(3):
        fn()
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(it):
        o = fn()
    jax.block_until_ready(o)
    return (time.perf_counter() - t0) / it * 1e3


print(f"# DeepSeek-R1 combine A/B: H={H} K={K} EP={EP}\n")
print("## CORRECTNESS (max rel-err vs numpy; carry & indep should be ~1e-3, "
      "naive-indep expected large on sparse)")
for pack in ("first", "scattered"):
    T = 4096
    idx, wts, valid, xnp = build(T, 7, pack)
    xc = jnp.asarray(xnp).astype(jnp.bfloat16)
    ic, wc, vc = jnp.asarray(idx), jnp.asarray(wts), jnp.asarray(valid)
    ref = ref_np(idx, wts, valid, xnp, T)
    e_carry = relerr(
        np.asarray(carry(xc, ic, wc, vc, reduce_group_size=K).astype(
            jnp.float32))[:T], ref)
    e_indep = relerr(
        np.asarray(indep_combine(xc, ic, wc, vc, K).astype(jnp.float32))[:T],
        ref)
    e_naive = relerr(
        np.asarray(naive_indep(xc, ic, wc, vc, reduce_group_size=K).astype(
            jnp.float32))[:T], ref)
    vps = valid.reshape(T, K).sum(1)
    print(f"  [{pack:9s}] valid/tok min={vps.min()} max={vps.max()} "
          f"mean={vps.mean():.1f} | carry={e_carry:.2e}  indep={e_indep:.2e}  "
          f"naive_indep={e_naive:.2e}")

print("\n## TIMING (valid-first; E2E ms; indep-E2E includes the preprocess)")
for T in (512, 4096, 16384, 65536):
    idx, wts, valid, xnp = build(T, 11, "first")
    xc = jnp.asarray(xnp).astype(jnp.bfloat16)
    ic, wc, vc = jnp.asarray(idx), jnp.asarray(wts), jnp.asarray(valid)
    t_carry = wall(lambda: carry(xc, ic, wc, vc, reduce_group_size=K))
    t_indep = wall(lambda: indep_combine(xc, ic, wc, vc, K))
    win = 100 * (t_carry - t_indep) / t_carry
    print(f"  T={T:6d}  carry={t_carry:7.3f}ms  indep_E2E={t_indep:7.3f}ms  "
          f"WIN={win:+5.1f}%")

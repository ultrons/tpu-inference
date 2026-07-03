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
"""AOT Mosaic compile check for the `indep` SparseCore combine kernel.

Compiles the indep kernel for a virtual tpu7x:2x2x1 topology — NO physical TPU
needed (libtpu must be in the venv). Catches Mosaic shape-cast / relayout errors
in `main_kernel` (the same kernel run_prepacked/indep_combine use) locally, in
seconds, before any cluster run.

    python tests/kernels/aot_indep_combine.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import topologies

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_indep import \
    ragged_gather_reduce as indep_kernel


def run_aot_compile_check():
    # tpu7x:2x2x1 = 4 chips × 2 cores = 8 virtual devices; Mosaic compiles with
    # the real v7x backend. No data is allocated (ShapeDtypeStruct only).
    topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")

    T, K, H = 2048, 8, 7168  # DeepSeek-R1 hidden; large enough to take the SC path
    P = T * K
    x = jax.ShapeDtypeStruct((P, H), jnp.bfloat16)
    idx = jax.ShapeDtypeStruct((P,), jnp.int32)
    w = jax.ShapeDtypeStruct((P,), jnp.float32)
    valid = jax.ShapeDtypeStruct((P,), jnp.bool_)

    with jax.default_device(topo.devices[0]):
        lowered = indep_kernel.lower(x, idx, w, valid, reduce_group_size=K)
        lowered.compile()  # Mosaic compiles for real; raises on any shape error
    print("AOT OK: indep SparseCore combine kernel compiled for tpu7x:2x2x1 "
          f"(P={P}, H={H}, K={K})")


if __name__ == "__main__":
    run_aot_compile_check()

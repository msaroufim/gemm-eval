# Skills for Writing Fast NVFP4 GEMM Kernels on Blackwell (SM100)

This document captures key insights from a kernel competition where developers wrote highly optimized NVFP4 (4-bit floating point) GEMM kernels for NVIDIA B200 GPUs. The focus is on **CuTe DSL patterns**, the preferred approach for writing maintainable high-performance kernels.

---

## 1. Understanding the Problem

### What is NVFP4 Block-Scaled GEMM?

NVFP4 is NVIDIA's 4-bit floating point format (E2M1 - 2 exponent bits, 1 mantissa bit). Because 4-bit precision has very limited dynamic range, **block scaling** is used: every 16 consecutive elements along the K dimension share a single FP8 scale factor.

```
Inputs:
  A:   (M, K, L)     - FP4 (Float4E2M1FN)
  B:   (N, K, L)     - FP4 (Float4E2M1FN)
  SFA: (M, K/16, L)  - FP8 (Float8E4M3FN) scale factors for A
  SFB: (N, K/16, L)  - FP8 (Float8E4M3FN) scale factors for B

Output:
  C:   (M, N, L)     - FP16

Computation: C[i,j] = sum_k(A[i,k] * SFA[i,k//16] * B[j,k] * SFB[j,k//16])
```

The tensor core hardware performs dequantization implicitly during the operation.

---

## 2. CuTe DSL Fundamentals

CuTe (CUDA Templates) is NVIDIA's DSL for expressing tensor operations with automatic layout optimization.

### 2.1 Core Imports

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr
import cutlass.pipeline as pipeline
```

### 2.2 Data Types

```python
ab_dtype = cutlass.Float4E2M1FN    # 4-bit FP for A and B matrices
sf_dtype = cutlass.Float8E4M3FN    # 8-bit FP for scale factors
c_dtype = cutlass.Float16          # 16-bit FP for output
acc_dtype = cutlass.Float32        # 32-bit FP for accumulators (in TMEM)
sf_vec_size = 16                   # 16 elements per scale factor block
```

### 2.3 Key Tile Parameters

```python
mma_tiler_mnk = (128, 128, 256)  # Tile shape for MMA operations
mma_inst_shape_k = 64            # K dimension per MMA instruction
threads_per_cta = 128            # Threads per CTA (4 warps)
num_ab_stage = 1                 # Number of pipeline stages for A/B
num_acc_stage = 1                # Number of pipeline stages for accumulator
num_tmem_alloc_cols = 512        # TMEM columns to allocate
```

---

## 3. Creating Tensors

### 3.1 Input Tensors with Layout Hints

Use `cute.assume()` for alignment hints that enable better optimization:

```python
# A tensor: (M, K, L) with K-major layout
a_tensor = cute.make_tensor(
    a_ptr,
    cute.make_layout(
        (m, cute.assume(k, 32), l),  # K is 32-aligned
        stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
    ),
)

# Alternative: ordered layout (specifies memory order)
a_tensor = cute.make_tensor(
    a_ptr,
    cute.make_ordered_layout(
        (cute.assume(m, 32), k, l),
        order=(1, 0, 2)  # K-major ordering
    ),
)
```

### 3.2 Scale Factor Tensor Layout

Scale factors require a special layout transformation for hardware compatibility:

```python
# Use blockscaled_utils to create the correct layout
sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
    a_tensor.shape,
    sf_vec_size  # 16
)
sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
```

This transforms from `(M, K/16, L)` to the hierarchical format `((Atom_M, Rest_M), (Atom_K, Rest_K), RestL)` where `Atom_M = (32, 4)` and `Atom_K = 4`.

---

## 4. TiledMMA Setup

### 4.1 Creating the MMA Operation

```python
# Create the MMA operation for NVFP4
mma_op = tcgen05.MmaMXF4NVF4Op(
    sf_dtype,                              # Scale factor type (Float8E4M3FN)
    (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),  # (128, 128, 64)
    tcgen05.CtaGroup.ONE,                  # CTA group size
    tcgen05.OperandSource.SMEM,            # Operands come from shared memory
)
tiled_mma = cute.make_tiled_mma(mma_op)
```

### 4.2 Higher-Level Tiled MMA (with scale factors)

```python
tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
    ab_dtype,                           # Float4E2M1FN
    tcgen05.OperandMajorMode.K,         # A is K-major
    tcgen05.OperandMajorMode.K,         # B is K-major
    sf_dtype,                           # Float8E4M3FN
    sf_vec_size,                        # 16
    tcgen05.CtaGroup.ONE,
    mma_tiler_mn,                       # (128, 128)
)
```

---

## 5. Shared Memory Layouts

### 5.1 Creating SMEM Layouts for A/B

```python
# Use sm100_utils to create optimal SMEM layouts
a_smem_layout_staged = sm100_utils.make_smem_layout_a(
    tiled_mma,
    mma_tiler_mnk,
    ab_dtype,
    num_ab_stage,
)

b_smem_layout_staged = sm100_utils.make_smem_layout_b(
    tiled_mma,
    mma_tiler_mnk,
    ab_dtype,
    num_ab_stage,
)
```

### 5.2 Creating SMEM Layouts for Scale Factors

```python
sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
    tiled_mma,
    mma_tiler_mnk,
    sf_vec_size,
    num_ab_stage,
)

sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
    tiled_mma,
    mma_tiler_mnk,
    sf_vec_size,
    num_ab_stage,
)
```

### 5.3 Allocating Shared Memory in Kernel

```python
@cute.kernel
def kernel(...):
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Allocate tensors with swizzling
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,  # Swizzle for bank conflict avoidance
    )

    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )

    # Scale factors don't need swizzling
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
```

---

## 6. TMA (Tensor Memory Accelerator) Setup

### 6.1 Creating TMA Atoms

```python
# TMA atom for loading A from global to shared memory
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),  # G2S operation
    a_tensor,                                                # Source tensor
    a_smem_layout,                                           # Target layout (without stage dim)
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
)

# TMA atom for loading B
tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    b_tensor,
    b_smem_layout,
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
)

# TMA for scale factors (note: internal_type=Int16 for correct addressing)
tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    sfa_tensor,
    sfa_smem_layout,
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
    internal_type=cutlass.Int16,  # Important for scale factors!
)
```

### 6.2 TMA Partitioning

```python
# Partition global and shared tensors for TMA
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a,
    0,                           # Coordinate in cluster
    cute.make_layout(1),         # CTA layout
    cute.group_modes(sA, 0, 3),  # Shared memory tensor
    cute.group_modes(tCgA, 0, 3), # Global memory tensor
)
```

### 6.3 Using TMA in the Kernel

```python
# Issue TMA load
cute.copy(
    tma_atom_a,
    tAgA[(None, k_tile)],        # Source (global memory slice)
    tAsA[(None, stage_index)],   # Destination (shared memory)
    tma_bar_ptr=ab_empty.barrier, # Barrier for completion tracking
)
```

---

## 7. TMEM (Tensor Memory) Management

### 7.1 Allocating TMEM

```python
# Create named barrier for TMEM allocation sync
tmem_alloc_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=threads_per_cta,
)

# Create TMEM allocator
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf,           # Shared memory for coordination
    barrier_for_retrieve=tmem_alloc_barrier,
)

# Allocate columns
tmem.allocate(num_tmem_alloc_cols)  # e.g., 512
tmem.wait_for_alloc()

# Get pointer to accumulator region
acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
```

### 7.2 Scale Factor TMEM Layout

```python
# SFA goes after accumulator in TMEM
sfa_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
    dtype=sf_dtype,
)

tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
    tiled_mma,
    mma_tiler_mnk,
    sf_vec_size,
    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
)
tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

# SFB goes after SFA
sfb_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr
    + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
    + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
    dtype=sf_dtype,
)
tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
```

### 7.3 Deallocating TMEM

```python
# At end of kernel
cute.arch.barrier()  # Sync all threads
tmem.free(acc_tmem_ptr)
```

---

## 8. Pipeline Management

### 8.1 Creating Pipelines

```python
# Producer-consumer groups
ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)

# TMA pipeline (for A/B loads)
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    num_stages=num_ab_stage,
    producer_group=ab_pipeline_producer_group,
    consumer_group=ab_pipeline_consumer_group,
    tx_count=num_tma_load_bytes,
).make_participants()

# Accumulator pipeline (for MMA -> epilogue)
acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    num_stages=num_acc_stage,
    producer_group=ab_pipeline_producer_group,
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
).make_participants()
```

### 8.2 Using Pipeline in Mainloop

```python
# Wait for accumulator buffer to be empty
acc_empty = acc_producer.acquire_and_advance()

# Disable accumulation for first iteration
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

for k_tile in range(k_tile_cnt):
    # Wait for AB buffer empty (producer side)
    ab_empty = ab_producer.acquire_and_advance()

    # Issue TMA loads
    cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)],
              tma_bar_ptr=ab_empty.barrier)
    cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, ab_empty.index)],
              tma_bar_ptr=ab_empty.barrier)
    # ... load SFA, SFB similarly

    # Wait for AB buffer full (consumer side)
    ab_full = ab_consumer.wait_and_advance()

    # Copy scale factors to TMEM
    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

    # Execute MMA for each K block
    num_kblocks = cute.size(tCrA, mode=[2])
    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
        # Set scale factor tensors
        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)

        # Execute GEMM
        cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)

        # Enable accumulation after first k block
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    # Release AB buffer
    ab_full.release()

acc_empty.commit()
```

---

## 9. SMEM-to-TMEM Copy (S2T)

### 9.1 Setting Up S2T Copy

```python
# Create S2T copy atom
copy_atom_s2t = cute.make_copy_atom(
    tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
    sf_dtype,
)

# Filter zeros from tensors (required for scale factors)
tCsSFA_compact = cute.filter_zeros(sSFA)
tCtSFA_compact = cute.filter_zeros(tCtSFA)

# Create tiled copy operation
tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)

# Partition source and destination
tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
    tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
)
tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
```

---

## 10. Epilogue: TMEM to Output

### 10.1 Setting Up TMEM-to-Register Copy

```python
# Create T2R (TMEM to Register) copy operation
op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)

# Create tiled copy from accumulator
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

# Partition accumulator and output
tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
tTR_gC = thr_copy_t2r.partition_D(tCgC)

# Create register tensors
tTR_rAcc = cute.make_rmem_tensor(
    tTR_gC[None, None, None, None, 0, 0, 0].shape,
    cutlass.Float32
)
tTR_rC = cute.make_rmem_tensor(
    tTR_gC[None, None, None, None, 0, 0, 0].shape,
    c_dtype  # Float16
)
```

### 10.2 Executing Epilogue

```python
# Wait for accumulator to be ready
acc_full = acc_consumer.wait_and_advance()

# Copy accumulator from TMEM to registers
cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

# Convert FP32 to FP16
acc_vec = tTR_rAcc.load().to(c_dtype)
tTR_rC.store(acc_vec)

# Store to global memory
simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
cute.copy(simt_atom, tTR_rC, tTR_gC)

acc_full.release()
```

---

## 11. Warp Specialization Pattern

For higher performance, use warp specialization with dedicated warps for TMA, MMA, and epilogue:

### 11.1 Warp Role Assignment

```python
class Kernel:
    epilog_warp_id = (0, 1, 2, 3)  # 4 warps for epilogue
    mma_warp_id = 4                 # 1 warp for MMA
    tma_warp_id = 5                 # 1 warp for TMA
    threads_per_cta = 32 * 6        # 192 threads total
```

### 11.2 Warp-Specialized Kernel Structure

```python
@cute.kernel
def kernel(self, ...):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    if warp_idx == self.tma_warp_id:
        # TMA Producer: Prefetch and load data
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        # ... issue TMA loads

    elif warp_idx == self.mma_warp_id:
        # MMA Producer: Execute matrix multiply
        for k_tile in range(k_tile_cnt):
            ab_full = ab_consumer.wait_and_advance()
            # Copy scale factors to TMEM
            # Execute cute.gemm()
            ab_full.release()

    elif warp_idx in self.epilog_warp_id:
        # Epilogue Consumers: Write results
        acc_full = acc_consumer.wait_and_advance()
        # Copy from TMEM to registers
        # Convert and store to global memory
        acc_full.release()
```

---

## 12. Cluster Support

### 12.1 Cluster Layout

```python
cluster_shape_mn = (2, 1)  # 2 CTAs in M, 1 in N

cluster_layout_vmnk = cute.tiled_divide(
    cute.make_layout((*cluster_shape_mn, 1)),
    (tiled_mma.thr_id.shape,),
)
```

### 12.2 Multicast Masks for TMA

```python
a_full_mcast_mask = cpasync.create_tma_multicast_mask(
    cluster_layout_vmnk,
    block_in_cluster_coord_vmnk,
    mcast_mode=2  # Multicast along M dimension
)

b_full_mcast_mask = cpasync.create_tma_multicast_mask(
    cluster_layout_vmnk,
    block_in_cluster_coord_vmnk,
    mcast_mode=1  # Multicast along N dimension
)
```

### 12.3 Cluster Synchronization

```python
cute.arch.cluster_arrive_relaxed()  # Non-blocking arrive
cute.arch.cluster_wait()            # Wait for all CTAs in cluster
```

---

## 13. Kernel Launch

### 13.1 JIT Function Structure

```python
@cute.jit
def my_kernel(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, problem_size):
    m, n, k, l = problem_size

    # Setup tensors, TMA atoms, layouts...

    # Compute grid size
    grid = (
        cute.ceil_div(m, mma_tiler_mnk[0]),
        cute.ceil_div(n, mma_tiler_mnk[1]),
        l,
    )

    # Launch kernel
    kernel(...).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),  # or cluster_shape for multi-SM
    )
```

### 13.2 Pre-compilation for Performance

```python
# Compile once with dummy pointers
def compile_kernel():
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    # ...
    return cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0,0,0,0))

# Execute with real data
compiled_kernel = compile_kernel()
compiled_kernel(real_a_ptr, real_b_ptr, ..., (m, n, k, l))
```

---

## 14. Scale Factor Pre-Processing (CPU Side)

The hardware requires scale factors in a specific permuted layout. Pre-process on CPU:

```python
def permute_scale_factors(sf, m, k, l, sf_vec_size=16):
    """
    Transform scale factors from (M, K/16, L) to hardware layout.
    """
    rest_m = m // 128
    rest_k = (k // sf_vec_size) // 4

    # Reshape to hierarchical format
    sf = sf.view(l, rest_m, 32, 4, rest_k, 4)

    # Permute to: (32, 4, rest_m, 4, rest_k, l)
    sf = sf.permute(2, 3, 1, 5, 4, 0).contiguous()

    return sf
```

---

## 15. Complete Minimal Example

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr
import cutlass.pipeline as pipeline
import cutlass.utils as utils

# Configuration
mma_tiler_mnk = (128, 128, 256)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16

@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a, mA, tma_atom_b, mB,
    tma_atom_sfa, mSFA, tma_atom_sfb, mSFB,
    mC,
    a_smem_layout_staged, b_smem_layout_staged,
    sfa_smem_layout_staged, sfb_smem_layout_staged,
    num_tma_load_bytes,
):
    tidx = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    # Allocate shared memory
    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(ab_dtype, a_smem_layout_staged.outer, 128,
                               swizzle=a_smem_layout_staged.inner)
    sB = smem.allocate_tensor(ab_dtype, b_smem_layout_staged.outer, 128,
                               swizzle=b_smem_layout_staged.inner)
    sSFA = smem.allocate_tensor(sf_dtype, sfa_smem_layout_staged, 128)
    sSFB = smem.allocate_tensor(sf_dtype, sfb_smem_layout_staged, 128)

    # Setup pipeline
    # ... (pipeline initialization)

    # Allocate TMEM
    tmem = utils.TmemAllocator(...)
    tmem.allocate(512)
    acc_ptr = tmem.retrieve_ptr(cutlass.Float32)

    # Local tile partitioning
    gA = cute.local_tile(mA, cute.slice_(mma_tiler_mnk, (None, 0, None)), ...)
    gB = cute.local_tile(mB, cute.slice_(mma_tiler_mnk, (0, None, None)), ...)
    k_tile_cnt = cute.size(gA, mode=[3])

    # Mainloop
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
    for k_tile in range(k_tile_cnt):
        # TMA load A, B, SFA, SFB
        cute.copy(tma_atom_a, gA_slice, sA_slice, tma_bar_ptr=barrier)
        # ... more copies

        # Wait for data
        # ... pipeline wait

        # Copy scale factors to TMEM
        cute.copy(s2t_copy, sSFA_slice, tCtSFA)
        cute.copy(s2t_copy, sSFB_slice, tCtSFB)

        # Execute GEMM
        for kblock in range(num_kblocks):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[kblock].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[kblock].iterator)
            cute.gemm(tiled_mma, tCtAcc, sA[kblock], sB[kblock], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    # Epilogue: TMEM -> registers -> global
    cute.copy(t2r_copy, tCtAcc, rAcc)
    rC = rAcc.load().to(c_dtype)
    cute.copy(stg_atom, rC, gC)

    tmem.free(acc_ptr)
```

---

## 16. Tile Size Selection Guide

| Workload | BLOCK_M | BLOCK_N | BLOCK_K | Stages | Notes |
|----------|---------|---------|---------|--------|-------|
| Large K (16K+) | 128 | 128 | 256 | 6-8 | Compute bound |
| Medium K (4-8K) | 128 | 64 | 256 | 5-8 | Balanced |
| Small K (2K) | 128 | 64 | 256 | 8 | Memory bound |
| Small N | 128 | 64 | 512 | 5 | Reduce N tile |

---

## 17. Common Pitfalls

1. **Forgetting `internal_type=cutlass.Int16`** for scale factor TMA atoms
2. **Not calling `cute.filter_zeros()`** on scale factor tensors before S2T copy
3. **Missing `tmem.wait_for_alloc()`** before using TMEM
4. **Forgetting to set `ACCUMULATE` field** to False for first iteration
5. **Not deallocating TMEM** at kernel end
6. **Wrong alignment** for `make_ptr()` - use 16 for data, 32 for scale factors

---

## 18. Performance Tips

1. **Use warp specialization** for overlapped execution
2. **Prefetch TMA descriptors** at kernel start
3. **Pre-compile kernels** to avoid JIT overhead during timing
4. **Use cluster multicast** when matrix dimensions allow
5. **Tune stage counts** based on SMEM capacity
6. **Pre-permute scale factors** on CPU to avoid GPU overhead

---

## Summary

Writing fast NVFP4 kernels with CuTe DSL involves:

1. **Tensor setup**: Use `cute.make_tensor()` with alignment hints
2. **TiledMMA**: Use `sm100_utils.make_blockscaled_trivial_tiled_mma()`
3. **TMA atoms**: Use `cute.nvgpu.make_tiled_tma_atom_A/B()` with correct internal types
4. **SMEM layouts**: Use `sm100_utils.make_smem_layout_*()` and `blockscaled_utils.make_smem_layout_sf*()`
5. **TMEM management**: Use `utils.TmemAllocator` with proper barriers
6. **Pipelines**: Use `pipeline.PipelineTmaUmma` for producer-consumer sync
7. **S2T copy**: Use `tcgen05.make_s2t_copy()` with `cute.filter_zeros()`
8. **Epilogue**: Use `tcgen05.make_tmem_copy()` for TMEM→register→global

The CuTe DSL abstracts away low-level PTX while still enabling near-peak performance through proper layout optimization and hardware feature utilization.

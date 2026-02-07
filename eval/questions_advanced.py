"""
Additional questions extracted from DeepGEMM and SonicMoE expert implementations.
"""

ADVANCED_QUESTIONS = {
    "tma_advanced": {
        "name": "TMA Advanced Patterns",
        "description": "Advanced TMA patterns from DeepGEMM/SonicMoE",
        "questions": [
            {
                "id": "tma_adv_1",
                "difficulty": "hard",
                "question": "In MoE GEMM, why is CPASYNC preferred over TMA for loading the activation matrix (B)?",
                "choices": [
                    "A) CPASYNC is faster than TMA",
                    "B) Different experts have different token counts, requiring frequent TMA descriptor updates which are expensive",
                    "C) TMA doesn't support activation matrices",
                    "D) CPASYNC uses less shared memory"
                ],
                "correct": "B",
                "explanation": "In MoE decoding, token counts per expert vary. TMA descriptors encode tensor shape, so variable shapes require expensive descriptor updates. CPASYNC loads directly without descriptors.",
                "reference": "cutlass/examples/92_blackwell_moe_gemm/ - Mixed TMA+CPASYNC strategy"
            },
            {
                "id": "tma_adv_2",
                "difficulty": "hard",
                "question": "What does cta_group::2 mean in a TMA instruction on Blackwell?",
                "choices": [
                    "A) Use 2 threads for the copy",
                    "B) Copy 2x the data",
                    "C) Synchronize across 2 CTAs in a cluster (2-SM mode)",
                    "D) Use 2 TMA units"
                ],
                "correct": "C",
                "explanation": "cta_group::2 indicates cluster execution mode spanning 2 SMs. This enables synchronized communication across CTAs and is used with 2x2x1 cluster configurations.",
                "reference": "cutlass/include/cute/arch/copy_sm100_tma.hpp"
            },
            {
                "id": "tma_adv_3",
                "difficulty": "medium",
                "question": "What is the size of a TMA descriptor (tensormap) in bytes?",
                "choices": [
                    "A) 32 bytes",
                    "B) 64 bytes",
                    "C) 128 bytes",
                    "D) 256 bytes"
                ],
                "correct": "C",
                "explanation": "TMA descriptors are 128 bytes and encode tensor shape, strides, data type, and addressing mode. They are stored in shared memory and updated via SMEM management.",
                "reference": "sonic-moe/functional/grouped_gemm.py - bytes_per_tensormap = 128"
            }
        ]
    },

    "umma": {
        "name": "UMMA (Unified MMA)",
        "description": "Blackwell's unified matrix multiply accumulate",
        "questions": [
            {
                "id": "umma_1",
                "difficulty": "medium",
                "question": "What are the two main operand modes for tcgen05 UMMA on Blackwell?",
                "choices": [
                    "A) SS (Smem-Smem) and RS (Register-Smem)",
                    "B) SS (Smem-Smem) and TS (TMEM-Smem)",
                    "C) RR (Register-Register) and SR (Smem-Register)",
                    "D) GS (Global-Smem) and SG (Smem-Global)"
                ],
                "correct": "B",
                "explanation": "SS mode: Both A and B from shared memory via TMA descriptors. TS mode: A from TMEM (for chained operations), B from shared memory. This enables efficient matrix chains.",
                "reference": "cutlass/include/cute/arch/mma_sm100_umma.hpp"
            },
            {
                "id": "umma_2",
                "difficulty": "hard",
                "question": "What is the UMMA::ScaleOut parameter used for?",
                "choices": [
                    "A) Scaling the input matrices",
                    "B) ScaleOut::Zero clears accumulator (first iteration), ScaleOut::One accumulates onto previous results",
                    "C) Scaling the output matrix",
                    "D) Setting the output precision"
                ],
                "correct": "B",
                "explanation": "ScaleOut controls accumulation behavior: Zero (0) initializes accumulator to zero before MMA, One (1) adds to existing accumulator. Critical for correct multi-tile GEMM.",
                "reference": "cutlass/include/cute/arch/mma_sm100_umma.hpp - ScaleOut enum"
            },
            {
                "id": "umma_3",
                "difficulty": "medium",
                "question": "What tile shapes are supported for M dimension in tcgen05 UMMA?",
                "choices": [
                    "A) Any multiple of 16",
                    "B) 64 or 128 only",
                    "C) 32, 64, 128, or 256",
                    "D) Only 128"
                ],
                "correct": "B",
                "explanation": "tcgen05 UMMA supports M=64 or M=128 for the tile shape. N can be 8-256 in multiples of 8. These fixed sizes match the tensor core hardware.",
                "reference": "cutlass/include/cute/arch/mma_sm100_umma.hpp - Tile shapes"
            }
        ]
    },

    "tmem_advanced": {
        "name": "TMEM Advanced",
        "description": "Advanced TMEM patterns from Blackwell",
        "questions": [
            {
                "id": "tmem_adv_1",
                "difficulty": "hard",
                "question": "What is the total TMEM capacity per SM on Blackwell?",
                "choices": [
                    "A) 256 KB",
                    "B) 512 KB",
                    "C) 128 DP × 512 columns × 32 bits = ~8 MB addressable",
                    "D) Same as shared memory (232 KB)"
                ],
                "correct": "C",
                "explanation": "TMEM has 128 Data Partitions (DP) × 512 columns with 32-bit addressing. This provides substantial accumulator storage separate from shared memory.",
                "reference": "cutlass/include/cute/arch/tmem_allocator_sm100.hpp - MAX_CAPACITY_BITS"
            },
            {
                "id": "tmem_adv_2",
                "difficulty": "medium",
                "question": "What is the minimum allocation granularity for TMEM?",
                "choices": [
                    "A) 1 column",
                    "B) 8 columns",
                    "C) 32 columns per allocation slice",
                    "D) 128 columns"
                ],
                "correct": "C",
                "explanation": "TMEM allocates in 32-column slices with power-of-2 alignment. This is handled by the tcgen05.alloc instruction.",
                "reference": "cutlass/include/cute/arch/tmem_allocator_sm100.hpp - ColumnsPerAllocationSlice = 32"
            },
            {
                "id": "tmem_adv_3",
                "difficulty": "hard",
                "question": "What copy atoms are used for TMEM to Register (T2R) and Register to TMEM (R2T)?",
                "choices": [
                    "A) Standard ld.shared and st.shared",
                    "B) SM100_TMEM_LOAD_16dp256b8x (T2R) and SM100_TMEM_STORE_16dp128b16x (R2T)",
                    "C) TMA bulk copy",
                    "D) Warp shuffle instructions"
                ],
                "correct": "B",
                "explanation": "Dedicated TMEM copy atoms handle the data movement: 256-bit loads (T2R) for reading accumulators, 128-bit stores (R2T) for writing. These are specialized PTX instructions.",
                "reference": "cutlass/examples/112_blackwell_ssd - CopyAtomT2R/CopyAtomR2T usage"
            }
        ]
    },

    "wgmma": {
        "name": "WGMMA (Warp Group MMA)",
        "description": "128-thread warp group matrix operations",
        "questions": [
            {
                "id": "wgmma_1",
                "difficulty": "easy",
                "question": "How many threads are in a warp group for WGMMA operations?",
                "choices": [
                    "A) 32 threads (1 warp)",
                    "B) 64 threads (2 warps)",
                    "C) 128 threads (4 warps)",
                    "D) 256 threads (8 warps)"
                ],
                "correct": "C",
                "explanation": "WGMMA (Warp-Group Matrix Multiply Accumulate) uses 128 threads (4 warps) to maximize tensor core utilization on Hopper and Blackwell.",
                "reference": "sonic-moe/functional/grouped_gemm.py - 128 threads per warp group"
            },
            {
                "id": "wgmma_2",
                "difficulty": "medium",
                "question": "What is the benefit of pingpong double-buffering in WGMMA kernels?",
                "choices": [
                    "A) Reduces register usage",
                    "B) Uses 2x warp groups: one computes while one prefetches, overlapping compute and memory",
                    "C) Doubles the tile size",
                    "D) Reduces shared memory usage"
                ],
                "correct": "B",
                "explanation": "Pingpong uses 2 warp groups: while one executes MMA, the other prefetches next tiles. This overlaps compute and memory operations for higher throughput.",
                "reference": "sonic-moe/functional/grouped_gemm.py - pingpong mode"
            }
        ]
    },

    "quantization": {
        "name": "FP8/FP4 Quantization",
        "description": "Low-precision GEMM patterns",
        "questions": [
            {
                "id": "quant_1",
                "difficulty": "easy",
                "question": "What FP8 formats are supported on Blackwell?",
                "choices": [
                    "A) Only E5M2",
                    "B) E4M3 and E5M2",
                    "C) E3M4 and E4M3",
                    "D) Only E4M3"
                ],
                "correct": "B",
                "explanation": "Blackwell supports both E4M3 (higher precision) and E5M2 (higher range) FP8 formats. E4M3 is typically used for weights, E5M2 for gradients.",
                "reference": "cutlass/examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu"
            },
            {
                "id": "quant_2",
                "difficulty": "medium",
                "question": "What accumulation precision is used for FP8/FP4 GEMM to maintain accuracy?",
                "choices": [
                    "A) FP8 accumulation",
                    "B) FP16 accumulation",
                    "C) FP32 accumulation",
                    "D) INT32 accumulation"
                ],
                "correct": "C",
                "explanation": "FP8/FP4 inputs are accumulated in FP32 to prevent precision loss. The result is then optionally quantized back to FP8/FP16 with dynamic scaling.",
                "reference": "cutlass/examples/70_blackwell_gemm - ElementAccumulator = float"
            },
            {
                "id": "quant_3",
                "difficulty": "hard",
                "question": "In block-scaled FP4 GEMM (NVFP4), what is gran_k and how does it affect scale factors?",
                "choices": [
                    "A) Gran_k is the K tile size for computation",
                    "B) Gran_k is the number of K elements sharing one scale factor (granularity)",
                    "C) Gran_k is the gradient accumulation factor",
                    "D) Gran_k controls gradient checkpointing"
                ],
                "correct": "B",
                "explanation": "Gran_k (granularity along K) defines how many K elements share one scale factor. For NVFP4 with block16, gran_k=16 means every 16 K elements have one FP8 scale.",
                "reference": "DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp - gran_k_a, gran_k_b"
            }
        ]
    },

    "heuristics": {
        "name": "Block Size Heuristics",
        "description": "Configuration selection for SM100",
        "questions": [
            {
                "id": "heur_1",
                "difficulty": "medium",
                "question": "What is the shared memory capacity per CTA on Blackwell?",
                "choices": [
                    "A) 164 KB",
                    "B) 192 KB",
                    "C) 232 KB",
                    "D) 256 KB"
                ],
                "correct": "C",
                "explanation": "Blackwell provides 232 KB shared memory per CTA, up from Hopper's 228 KB. This enables larger tile sizes and more pipeline stages.",
                "reference": "DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp - smem_capacity = 232448"
            },
            {
                "id": "heur_2",
                "difficulty": "hard",
                "question": "Why does DeepGEMM limit block_m to 128 for 1D1D kernel with K-major B?",
                "choices": [
                    "A) Hardware limitation",
                    "B) Shared memory constraint",
                    "C) Larger block_m has low performance due to reduced TMA efficiency with K-major layout",
                    "D) Register pressure"
                ],
                "correct": "C",
                "explanation": "For 1D1D kernel type with K-major B layout, block_m > 128 leads to suboptimal TMA access patterns and lower performance.",
                "reference": "DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp - is_block_size_legal()"
            },
            {
                "id": "heur_3",
                "difficulty": "medium",
                "question": "What constraint must block_n satisfy for proper TMA swizzling?",
                "choices": [
                    "A) Must be power of 2",
                    "B) block_n × element_size must align to 64 bytes for MN-major layout",
                    "C) Must be less than 256",
                    "D) Must equal block_m"
                ],
                "correct": "B",
                "explanation": "TMA swizzle requires block_n × element_size to align to 64 bytes. For FP16 (2 bytes), block_n must be multiple of 32. This ensures conflict-free bank access.",
                "reference": "DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp - swizzle constraints"
            }
        ]
    },

    "moe_patterns": {
        "name": "MoE GEMM Patterns",
        "description": "Mixture of Experts optimization patterns",
        "questions": [
            {
                "id": "moe_1",
                "difficulty": "medium",
                "question": "What is the grouped GEMM pattern in MoE?",
                "choices": [
                    "A) Running one large GEMM for all experts",
                    "B) Multiple variable-sized GEMMs, one per expert, with different token counts",
                    "C) Batched GEMM with fixed sizes",
                    "D) Sequential expert computation"
                ],
                "correct": "B",
                "explanation": "Grouped GEMM runs multiple GEMMs where each expert handles different numbers of tokens. This requires variable-sized problem handling and efficient scheduling.",
                "reference": "sonic-moe/functional/grouped_gemm.py"
            },
            {
                "id": "moe_2",
                "difficulty": "hard",
                "question": "How does SonicMoE handle GLU activations efficiently?",
                "choices": [
                    "A) Separate kernel for activation",
                    "B) Fused in epilogue: output splits into gate and up, applies gate × sigmoid(up)",
                    "C) Pre-computed activation tables",
                    "D) Activation computed on CPU"
                ],
                "correct": "B",
                "explanation": "GLU (SwiGLU, GeGLU, etc.) is fused in the epilogue. The up-projection output (2×I) is split into gate and up components, then combined inline to produce (I) output.",
                "reference": "sonic-moe/functional/grouped_gemm.py - is_glu, y_epi_tile"
            },
            {
                "id": "moe_3",
                "difficulty": "medium",
                "question": "What does expert_frequency_offset represent in MoE?",
                "choices": [
                    "A) Learning rate per expert",
                    "B) Cumulative sum of token counts per expert, used for gather/scatter indexing",
                    "C) Memory offset for expert weights",
                    "D) Frequency of expert selection in training"
                ],
                "correct": "B",
                "explanation": "expert_frequency_offset is cumsum of token counts: [0, count_e0, count_e0+count_e1, ...]. Used to index into grouped tensors for scatter/gather operations.",
                "reference": "sonic-moe/functional/__init__.py - TC_topk_router_metadata"
            }
        ]
    },

    "cluster": {
        "name": "Cluster & Multi-SM",
        "description": "Multi-CTA cluster patterns",
        "questions": [
            {
                "id": "cluster_1",
                "difficulty": "medium",
                "question": "What is a CTA cluster on Blackwell?",
                "choices": [
                    "A) Group of warps within one SM",
                    "B) Group of 2-4 CTAs that can communicate via shared memory across SMs",
                    "C) CPU thread cluster",
                    "D) Memory bank cluster"
                ],
                "correct": "B",
                "explanation": "CTA clusters group 2-4 CTAs across SMs that share data via distributed shared memory. Common shapes: (2,1,1), (1,2,1), (2,2,1). Enables TMA multicast and synchronized execution.",
                "reference": "sonic-moe/functional/moe_config.py - cluster_shape"
            },
            {
                "id": "cluster_2",
                "difficulty": "hard",
                "question": "What is TMA multicast and when is it useful?",
                "choices": [
                    "A) Broadcasting to all threads in a warp",
                    "B) Single TMA load delivers data to multiple CTAs in cluster, avoiding redundant loads",
                    "C) Multicasting between GPUs",
                    "D) Duplicating data in L2 cache"
                ],
                "correct": "B",
                "explanation": "TMA multicast allows one TMA load to deliver the same data to multiple CTAs in a cluster. Useful when CTAs share input tiles (e.g., along K reduction dimension).",
                "reference": "cutlass/include/cute/arch/copy_sm100_tma.hpp - multicast support"
            }
        ]
    }
}


def get_advanced_questions():
    """Return flat list of all advanced questions."""
    all_q = []
    for skill_id, skill_data in ADVANCED_QUESTIONS.items():
        for q in skill_data["questions"]:
            q_copy = q.copy()
            q_copy["skill"] = skill_id
            q_copy["skill_name"] = skill_data["name"]
            all_q.append(q_copy)
    return all_q

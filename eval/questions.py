"""
NVFP4 GEMM Skill Evaluation Questions

Each skill has multiple choice questions extracted from expert kernels (1.py-6.py).
Questions are tagged with difficulty: easy, medium, hard
"""

QUESTIONS = {
    "tma": {
        "name": "TMA (Tensor Memory Accelerator)",
        "description": "Async bulk copies from global to shared memory",
        "questions": [
            {
                "id": "tma_easy_1",
                "difficulty": "easy",
                "question": "What does TMA stand for and what is its primary purpose in GEMM kernels?",
                "choices": [
                    "A) Tensor Memory Accelerator - hardware unit for async data transfer from global to shared memory",
                    "B) Thread Memory Access - software abstraction for thread-local storage",
                    "C) Tile Matrix Accumulator - hardware unit for matrix multiplication",
                    "D) Transfer Mode Async - compiler optimization for memory prefetching"
                ],
                "correct": "A",
                "explanation": "TMA (Tensor Memory Accelerator) is a hardware unit in Blackwell GPUs that performs asynchronous bulk data transfers from global memory to shared memory, with tensor-aware addressing.",
                "reference": "1.py:128-153 - TMA copy operations"
            },
            {
                "id": "tma_easy_2",
                "difficulty": "easy",
                "question": "Which instruction is used for 3D tensor TMA copies?",
                "choices": [
                    "A) cp.sync.tensor.3d",
                    "B) cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes",
                    "C) ld.global.async.3d",
                    "D) memcpy.async.tensor"
                ],
                "correct": "B",
                "explanation": "The full instruction includes: async (non-blocking), bulk (large transfer), tensor.3d (3D addressing), shared::cta (destination), global (source), mbarrier::complete_tx::bytes (barrier signaling).",
                "reference": "1.py:380-400"
            },
            {
                "id": "tma_medium_1",
                "difficulty": "medium",
                "question": "In TMA operations, what is the correct order of barrier operations for the producer (TMA warp)?",
                "choices": [
                    "A) arrive → wait → expect_tx",
                    "B) expect_tx → issue TMA → arrive",
                    "C) wait → expect_tx → issue TMA",
                    "D) issue TMA → expect_tx → arrive"
                ],
                "correct": "B",
                "explanation": "Producer must: 1) expect_tx to set expected bytes, 2) issue the TMA copy, 3) arrive to signal work is submitted. The consumer then waits.",
                "reference": "1.py:450-480 - Pipeline producer pattern"
            },
            {
                "id": "tma_medium_2",
                "difficulty": "medium",
                "question": "What are the three cache eviction policies used in the expert kernels, and when is EVICT_FIRST preferred?",
                "choices": [
                    "A) EVICT_NORMAL, EVICT_FIRST, EVICT_LAST - EVICT_FIRST keeps data in L1, use for frequently reused data",
                    "B) EVICT_NORMAL, EVICT_FIRST, EVICT_LAST - EVICT_FIRST immediately evicts, use for streaming data",
                    "C) CACHE_L1, CACHE_L2, CACHE_NONE - CACHE_L1 is always preferred",
                    "D) READ_ONCE, READ_MANY, WRITE_BACK - READ_MANY for reused data"
                ],
                "correct": "A",
                "explanation": "EVICT_FIRST (0x12F0...) keeps data in L1 cache longer, preferred for tiles that will be reused. EVICT_LAST evicts to L2, used for streaming/one-time data.",
                "reference": "1.py:89-91 - Cache policy constants"
            },
            {
                "id": "tma_hard_1",
                "difficulty": "hard",
                "question": "In cluster multicast TMA, how is the multicast mask created for a (1,2) cluster shape along the N dimension?",
                "choices": [
                    "A) mask = 0x3 (binary 11) to target both CTAs in cluster",
                    "B) Use cpasync.create_tma_multicast_mask with mcast_mode=2 based on block coordinates",
                    "C) Manually compute mask as (1 << cluster_id) for each target",
                    "D) Multicast is automatic, no mask needed"
                ],
                "correct": "B",
                "explanation": "The multicast mask is computed using create_tma_multicast_mask which takes cluster layout, block coordinates, and mcast_mode to determine which CTAs receive the same data.",
                "reference": "2.py:539-574 - Cluster multicast setup"
            }
        ]
    },

    "async_copy": {
        "name": "Async Copy & Barrier Synchronization",
        "description": "cp.async patterns with mbarrier coordination",
        "questions": [
            {
                "id": "async_easy_1",
                "difficulty": "easy",
                "question": "What is the purpose of mbarrier (memory barrier) in async copy operations?",
                "choices": [
                    "A) To prevent race conditions between CPU and GPU",
                    "B) To synchronize producer and consumer threads - producer signals completion, consumer waits",
                    "C) To flush GPU caches to main memory",
                    "D) To serialize all memory operations globally"
                ],
                "correct": "B",
                "explanation": "mbarrier coordinates async operations: producers signal when data is ready (arrive), consumers wait until the signal (wait). This enables overlap without races.",
                "reference": "1.py:420-450 - Barrier usage in pipeline"
            },
            {
                "id": "async_easy_2",
                "difficulty": "easy",
                "question": "What does expect_tx do in the async copy pattern?",
                "choices": [
                    "A) Transmits data to the next stage",
                    "B) Sets the expected number of bytes that will arrive at the barrier",
                    "C) Waits for a transmission to complete",
                    "D) Cancels a pending async operation"
                ],
                "correct": "B",
                "explanation": "expect_tx tells the barrier how many bytes to expect. The barrier only completes when all expected bytes have arrived via TMA copies.",
                "reference": "1.py:460-465"
            },
            {
                "id": "async_medium_1",
                "difficulty": "medium",
                "question": "In a double-buffered pipeline with 2 stages, what happens if the producer issues stage 1 before the consumer finishes reading stage 1 from the previous iteration?",
                "choices": [
                    "A) Hardware automatically stalls the producer",
                    "B) Data corruption - producer overwrites data consumer is reading",
                    "C) The barrier wait prevents this - producer must wait for consumer to release the stage",
                    "D) Compiler error - this pattern is not allowed"
                ],
                "correct": "C",
                "explanation": "Proper double-buffering uses barriers: producer waits for consumer to release a stage (via arrive) before reusing it. This prevents data races.",
                "reference": "1.py:500-530 - Pipeline stage management"
            },
            {
                "id": "async_hard_1",
                "difficulty": "hard",
                "question": "In the expert kernels, why are there separate pipelines for A/B data vs accumulator data?",
                "choices": [
                    "A) Hardware limitation - only one TMA unit available",
                    "B) Different producer/consumer groups and timing - TMA produces A/B, MMA consumes A/B and produces accum, epilogue consumes accum",
                    "C) Memory bandwidth optimization - interleave different data types",
                    "D) Accumulator doesn't need barriers"
                ],
                "correct": "B",
                "explanation": "A/B pipeline: TMA warp produces, MMA warp consumes. Accumulator pipeline: MMA warp produces (writes TMEM), epilogue warps consume. Different warp groups require separate synchronization.",
                "reference": "1.py:314-350 - Warp role assignments"
            }
        ]
    },

    "tcgen05": {
        "name": "tcgen05 (Tensor Core Gen 5)",
        "description": "Hardware MMA unit for NVFP4 operations",
        "questions": [
            {
                "id": "tcgen05_easy_1",
                "difficulty": "easy",
                "question": "What does tcgen05 stand for and what hardware does it run on?",
                "choices": [
                    "A) Tensor Core Generation 5 - runs on Blackwell (SM100) GPUs",
                    "B) Thread Compute Group 05 - software threading model",
                    "C) Tensor Compression Gen 05 - data compression unit",
                    "D) Tile Cache Gen 05 - L2 cache controller"
                ],
                "correct": "A",
                "explanation": "tcgen05 refers to the 5th generation Tensor Core architecture in Blackwell GPUs, supporting FP4 matrix operations with block scaling.",
                "reference": "1.py:128-153"
            },
            {
                "id": "tcgen05_easy_2",
                "difficulty": "easy",
                "question": "What are the three main tcgen05 operations used in NVFP4 GEMM?",
                "choices": [
                    "A) load, store, compute",
                    "B) tcgen05.cp (copy to TMEM), tcgen05.mma (matrix multiply), tcgen05.ld (load from TMEM)",
                    "C) tcgen05.alloc, tcgen05.free, tcgen05.sync",
                    "D) tcgen05.init, tcgen05.run, tcgen05.finish"
                ],
                "correct": "B",
                "explanation": "tcgen05.cp copies scale factors from SMEM to TMEM, tcgen05.mma executes FP4 matrix multiply with block scaling, tcgen05.ld loads results from TMEM to registers.",
                "reference": "1.py:128-153 - tcgen05 instruction examples"
            },
            {
                "id": "tcgen05_medium_1",
                "difficulty": "medium",
                "question": "In tcgen05.mma for NVFP4, what does 'block_scale.block16' mean?",
                "choices": [
                    "A) Process 16 blocks at a time",
                    "B) Every 16 consecutive K-dimension elements share one FP8 scale factor",
                    "C) Use 16-bit accumulation",
                    "D) Tile size is 16x16"
                ],
                "correct": "B",
                "explanation": "Block scaling with block16 means every 16 elements along K share one scale factor. This reduces memory for scales while maintaining reasonable precision.",
                "reference": "1.py:140-145 - MMA instruction format"
            },
            {
                "id": "tcgen05_medium_2",
                "difficulty": "medium",
                "question": "What is the correct MMA instruction format for NVFP4 with block scaling?",
                "choices": [
                    "A) tcgen05.mma.f16.f4.f4",
                    "B) tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
                    "C) tcgen05.gemm.fp4.scaled",
                    "D) mma.sync.aligned.m16n8k32.f16.f4.f4"
                ],
                "correct": "B",
                "explanation": "The full instruction specifies: cta_group (thread group), kind::mxf4nvf4 (FP4 matrix types), block_scale (use scale factors), block16 (16-element blocks).",
                "reference": "1.py:140-145"
            },
            {
                "id": "tcgen05_hard_1",
                "difficulty": "hard",
                "question": "Why must ACCUMULATE be set to False for the first MMA iteration?",
                "choices": [
                    "A) Hardware initialization requirement",
                    "B) First iteration initializes accumulators to zero; subsequent iterations accumulate onto previous results",
                    "C) Compiler optimization",
                    "D) Memory alignment requirement"
                ],
                "correct": "B",
                "explanation": "ACCUMULATE=False clears the accumulator to zero before the first MMA. ACCUMULATE=True adds to existing accumulator values. Using True on first iteration would add to garbage values.",
                "reference": "1.py:600-620 - MMA loop with accumulate flag"
            }
        ]
    },

    "tmem": {
        "name": "TMEM (Tensor Memory)",
        "description": "On-chip memory for accumulators and scale factors",
        "questions": [
            {
                "id": "tmem_easy_1",
                "difficulty": "easy",
                "question": "What is TMEM and what is stored there during NVFP4 GEMM?",
                "choices": [
                    "A) Thread Memory - local variables per thread",
                    "B) Tensor Memory - accumulators (FP32) and scale factors (FP8) for tcgen05",
                    "C) Texture Memory - read-only cached data",
                    "D) Tile Memory - intermediate computation results"
                ],
                "correct": "B",
                "explanation": "TMEM is fast on-chip memory dedicated to tensor cores. It holds FP32 accumulators and FP8 scale factors that tcgen05 accesses during MMA operations.",
                "reference": "1.py:200-220 - TMEM layout"
            },
            {
                "id": "tmem_easy_2",
                "difficulty": "easy",
                "question": "What must be called before using TMEM in the kernel?",
                "choices": [
                    "A) tmem.init()",
                    "B) tmem.alloc() followed by tmem.wait_for_alloc()",
                    "C) cudaMalloc for TMEM",
                    "D) Just use it, no allocation needed"
                ],
                "correct": "B",
                "explanation": "TMEM must be explicitly allocated with tmem.alloc() and the kernel must wait for allocation to complete with wait_for_alloc() before use.",
                "reference": "1.py:350-360 - TMEM allocation"
            },
            {
                "id": "tmem_medium_1",
                "difficulty": "medium",
                "question": "In the TMEM layout, what is the order of regions from lowest to highest address?",
                "choices": [
                    "A) SFA, SFB, Accumulators",
                    "B) Accumulators, SFA, SFB",
                    "C) SFB, SFA, Accumulators",
                    "D) Interleaved based on access pattern"
                ],
                "correct": "B",
                "explanation": "TMEM is laid out as: [Accumulators (FP32)] [SFA region (FP8)] [SFB region (FP8)]. Accumulators come first as they're the primary use case.",
                "reference": "1.py:200-220 - TMEM allocation order"
            },
            {
                "id": "tmem_hard_1",
                "difficulty": "hard",
                "question": "Why does SFB handling differ for N=192 vs N=64 tile sizes?",
                "choices": [
                    "A) Different hardware units handle each size",
                    "B) N=192 requires special striding (offset=2 for odd tiles) due to 192=64*3 not being power-of-2 aligned",
                    "C) N=64 doesn't need scale factors",
                    "D) Memory bandwidth optimization only"
                ],
                "correct": "B",
                "explanation": "192=64*3 creates alignment issues. The kernel uses offset=2 for odd MMA tile coordinates to correctly index into TMEM columns. N=64 alternates between 2 SFB regions.",
                "reference": "2.py:798-842 - SFB offset calculation"
            }
        ]
    },

    "scale_factors": {
        "name": "Block-Scaled FP4 Layout",
        "description": "Scale factor permutation and application",
        "questions": [
            {
                "id": "sf_easy_1",
                "difficulty": "easy",
                "question": "In NVFP4, what data type is used for scale factors?",
                "choices": [
                    "A) FP32",
                    "B) FP16",
                    "C) FP8 (E4M3 or E5M2)",
                    "D) INT8"
                ],
                "correct": "C",
                "explanation": "Scale factors are stored as FP8, providing a good balance between range (for scaling) and memory efficiency.",
                "reference": "1.py:80-85 - Data type definitions"
            },
            {
                "id": "sf_easy_2",
                "difficulty": "easy",
                "question": "What is the scale factor block size in NVFP4 GEMM?",
                "choices": [
                    "A) 4 elements per scale factor",
                    "B) 8 elements per scale factor",
                    "C) 16 elements per scale factor (along K dimension)",
                    "D) 32 elements per scale factor"
                ],
                "correct": "C",
                "explanation": "Every 16 consecutive elements along the K dimension share one FP8 scale factor. SFA shape is (M, K/16, L).",
                "reference": "1.py:70-75 - Scale factor dimensions"
            },
            {
                "id": "sf_medium_1",
                "difficulty": "medium",
                "question": "Why do scale factors need to be permuted before being used by tcgen05?",
                "choices": [
                    "A) Compiler requirement",
                    "B) tcgen05 hardware expects a specific hierarchical layout matching its internal warp/thread organization",
                    "C) Memory coalescing optimization",
                    "D) They don't need permutation"
                ],
                "correct": "B",
                "explanation": "tcgen05 has specific expectations for how scale factors are laid out in memory to match its warp-level execution pattern. CPU-side permutation transforms (L,M,K/16) to the required hierarchical format.",
                "reference": "Reference kernels - permute_scale_factors function"
            },
            {
                "id": "sf_hard_1",
                "difficulty": "hard",
                "question": "What is the correct permutation for scale factors from (L, M, K/16) to tcgen05 layout?",
                "choices": [
                    "A) Simple transpose to (K/16, M, L)",
                    "B) Reshape to (L, M/128, 32, 4, K/64, 4) then permute dims (2,3,1,5,4,0)",
                    "C) Reshape to (L, M, K/16, 1) and broadcast",
                    "D) No reshape needed, just reinterpret"
                ],
                "correct": "B",
                "explanation": "The hierarchical reshape breaks M into (M/128, 32, 4) and K/16 into (K/64, 4), then permutes to match tcgen05's warp-level access pattern.",
                "reference": "Reference kernels - permute_scale_factors implementation"
            }
        ]
    },

    "warp_specialization": {
        "name": "Warp Specialization",
        "description": "Assigning different roles to different warps",
        "questions": [
            {
                "id": "warp_easy_1",
                "difficulty": "easy",
                "question": "What is warp specialization in GPU kernels?",
                "choices": [
                    "A) Making all warps do the same work",
                    "B) Assigning different roles (TMA, MMA, epilogue) to different warps for overlapped execution",
                    "C) Specializing warps for different data types",
                    "D) Hardware feature that auto-assigns work"
                ],
                "correct": "B",
                "explanation": "Warp specialization assigns different tasks to different warps: e.g., one warp issues TMA loads while another executes MMA, enabling overlap.",
                "reference": "1.py:314-350 - Warp role definitions"
            },
            {
                "id": "warp_medium_1",
                "difficulty": "medium",
                "question": "In the expert NVFP4 kernels, how are 6 warps (192 threads) typically assigned?",
                "choices": [
                    "A) All 6 warps do MMA",
                    "B) 4 warps epilogue, 1 warp MMA, 1 warp TMA",
                    "C) 3 warps TMA, 3 warps MMA",
                    "D) 2 warps each for TMA, MMA, epilogue"
                ],
                "correct": "B",
                "explanation": "Warps 0-3 (threads 0-127): epilogue (copy TMEM→registers→global). Warp 4 (threads 128-159): MMA execution. Warp 5 (threads 160-191): TMA loads.",
                "reference": "1.py:314-350 - Thread to warp mapping"
            },
            {
                "id": "warp_medium_2",
                "difficulty": "medium",
                "question": "Why are there more epilogue warps (4) than MMA warps (1)?",
                "choices": [
                    "A) Epilogue is more computationally intensive",
                    "B) MMA is pipelined and one warp can issue enough work; epilogue needs bandwidth to copy results from TMEM to global memory",
                    "C) Hardware limitation on MMA warps",
                    "D) Epilogue warps are idle most of the time"
                ],
                "correct": "B",
                "explanation": "MMA warp issues work to tensor cores which execute asynchronously. Epilogue needs multiple warps to provide enough memory bandwidth to move results from TMEM to global memory.",
                "reference": "1.py:314-350"
            },
            {
                "id": "warp_hard_1",
                "difficulty": "hard",
                "question": "How do specialized warps coordinate without excessive synchronization overhead?",
                "choices": [
                    "A) Global __syncthreads() after each stage",
                    "B) Producer-consumer pipelines with barriers - each warp only syncs with its partner(s)",
                    "C) Lock-free atomic operations",
                    "D) Separate kernel launches"
                ],
                "correct": "B",
                "explanation": "Pipelines use mbarriers for targeted synchronization: TMA warp signals MMA warp via ab_pipeline barrier, MMA warp signals epilogue warps via acc_pipeline barrier. No global sync needed.",
                "reference": "1.py:400-500 - Pipeline synchronization"
            }
        ]
    },

    "pipeline": {
        "name": "Pipeline Patterns",
        "description": "Double-buffering and multi-stage pipelines",
        "questions": [
            {
                "id": "pipe_easy_1",
                "difficulty": "easy",
                "question": "What is the purpose of double-buffering in GEMM kernels?",
                "choices": [
                    "A) Use two GPUs",
                    "B) Overlap data loading with computation - load next tile while computing current tile",
                    "C) Double the precision",
                    "D) Handle two matrices at once"
                ],
                "correct": "B",
                "explanation": "Double-buffering uses two memory slots: while MMA computes on slot 0, TMA loads into slot 1. Then they swap. This hides memory latency.",
                "reference": "1.py:450-500 - Pipeline staging"
            },
            {
                "id": "pipe_easy_2",
                "difficulty": "easy",
                "question": "What is a 'stage' in a software pipeline?",
                "choices": [
                    "A) A phase of compilation",
                    "B) A slot in the circular buffer where data resides, indexed by stage number",
                    "C) A GPU processing unit",
                    "D) A synchronization point"
                ],
                "correct": "B",
                "explanation": "Stages are slots in shared memory. With 4 stages, you have 4 buffer slots. Producer advances stage index after filling, consumer advances after reading.",
                "reference": "1.py:450-500"
            },
            {
                "id": "pipe_medium_1",
                "difficulty": "medium",
                "question": "How many pipeline stages are typically used for A/B data in the expert kernels?",
                "choices": [
                    "A) 2 stages (classic double-buffer)",
                    "B) 4-8 stages depending on K dimension",
                    "C) 1 stage (no buffering)",
                    "D) 16 stages always"
                ],
                "correct": "B",
                "explanation": "Larger K values use more stages (6-8) to keep TMA busy. Smaller K uses fewer stages (4-6). This is tuned based on memory latency vs shared memory capacity.",
                "reference": "1.py - num_ab_stage parameter"
            },
            {
                "id": "pipe_hard_1",
                "difficulty": "hard",
                "question": "What is 'overlapping accumulator' and why is it used?",
                "choices": [
                    "A) Using FP64 for higher precision",
                    "B) Two accumulator regions in TMEM - MMA writes to one while epilogue reads from the other, using phase bits to alternate",
                    "C) Accumulating across multiple kernels",
                    "D) CPU-GPU overlap for accumulation"
                ],
                "correct": "B",
                "explanation": "With overlapping_accum=True, there are 2 accumulator regions. Phase bits (XOR with index) alternate between them, allowing MMA and epilogue to work simultaneously on different data.",
                "reference": "2.py:784-836 - Overlapping accumulator pattern"
            }
        ]
    },

    "integration": {
        "name": "Integration & Debugging",
        "description": "Putting it all together correctly",
        "questions": [
            {
                "id": "int_easy_1",
                "difficulty": "easy",
                "question": "What is the correct data type for the TMA atom's internal_type when loading scale factors?",
                "choices": [
                    "A) cutlass.Float8",
                    "B) cutlass.Int16",
                    "C) cutlass.Int8",
                    "D) cutlass.Float16"
                ],
                "correct": "B",
                "explanation": "Scale factor TMA atoms use internal_type=cutlass.Int16 for proper alignment and transfer size, even though the actual data is FP8.",
                "reference": "skills.md pitfalls - internal_type requirement"
            },
            {
                "id": "int_medium_1",
                "difficulty": "medium",
                "question": "What function must be called on scale factor tensors before S2T (SMEM to TMEM) copy?",
                "choices": [
                    "A) cute.compact()",
                    "B) cute.filter_zeros()",
                    "C) cute.align()",
                    "D) cute.transpose()"
                ],
                "correct": "B",
                "explanation": "cute.filter_zeros() removes zero-stride modes from the tensor, creating a compact representation required by the S2T copy operation.",
                "reference": "2.py:858-867 - filter_zeros before S2T"
            },
            {
                "id": "int_medium_2",
                "difficulty": "medium",
                "question": "What alignment (in bytes) is required for scale factor pointers in make_ptr()?",
                "choices": [
                    "A) 4 bytes",
                    "B) 16 bytes (same as data)",
                    "C) 32 bytes",
                    "D) 128 bytes"
                ],
                "correct": "C",
                "explanation": "Data pointers use 16-byte alignment, but scale factor pointers require 32-byte alignment for correct tcgen05 operation.",
                "reference": "skills.md pitfalls - alignment requirements"
            },
            {
                "id": "int_hard_1",
                "difficulty": "hard",
                "question": "A kernel works for K=8192 but fails silently (wrong results) for K=256. What is the most likely cause?",
                "choices": [
                    "A) TMA descriptor misconfiguration",
                    "B) Pipeline depth too large - stages exceed K/tile_k iterations, causing uninitialized data use",
                    "C) Accumulator overflow",
                    "D) Scale factor underflow"
                ],
                "correct": "B",
                "explanation": "If num_stages > ceil(K/tile_k), the pipeline prefetches beyond the actual data. For small K, you need fewer stages or proper boundary checks.",
                "reference": "Expert kernel tuning for different K values"
            }
        ]
    }
}

def get_all_questions():
    """Return flat list of all questions with skill metadata."""
    all_q = []
    for skill_id, skill_data in QUESTIONS.items():
        for q in skill_data["questions"]:
            q_copy = q.copy()
            q_copy["skill"] = skill_id
            q_copy["skill_name"] = skill_data["name"]
            all_q.append(q_copy)
    return all_q

def get_questions_by_skill(skill_id):
    """Return questions for a specific skill."""
    if skill_id not in QUESTIONS:
        return []
    return QUESTIONS[skill_id]["questions"]

def get_questions_by_difficulty(difficulty):
    """Return all questions of a given difficulty."""
    return [q for q in get_all_questions() if q["difficulty"] == difficulty]

def get_skill_list():
    """Return list of skill IDs and names."""
    return [(k, v["name"]) for k, v in QUESTIONS.items()]

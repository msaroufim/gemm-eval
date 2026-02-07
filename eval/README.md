# NVFP4 GEMM AI Evaluation Framework

A granular evaluation framework to measure how well an AI understands NVFP4 GEMM implementation on NVIDIA Blackwell GPUs.

## Overview

This framework tests individual skills required to implement high-performance NVFP4 GEMM kernels:

| Category | Skills Tested | Questions |
|----------|---------------|-----------|
| **Core** | TMA, Async Copy, tcgen05, TMEM, Scale Factors, Warp Specialization, Pipelines, Integration | 34 |
| **Advanced** | TMA Advanced, UMMA, TMEM Advanced, WGMMA, Quantization, Heuristics, MoE Patterns, Clusters | 22 |
| **Code Challenges** | Scale factor permutation, barrier patterns, warp assignment, pipeline index, TMEM layout | 5 |

**Total: 56 multiple-choice questions + 5 runnable code challenges**

## Quick Start

```bash
cd eval

# List all available questions
python3 run_eval.py list

# Export prompts for batch evaluation (e.g., to send to an AI API)
python3 run_eval.py export -o prompts.json

# Run interactive evaluation (enter AI responses manually)
python3 run_eval.py interactive --model gpt-4

# Import responses and auto-score
python3 run_eval.py import -p prompts.json -r responses.json

# Manual annotation via TUI
python3 run_eval.py annotate -f results.json

# View report from saved results
python3 run_eval.py report -f results.json
```

## Skill Categories

### Core Skills (from 1.py-6.py expert kernels)

1. **TMA (Tensor Memory Accelerator)** - Async bulk copies, barriers, cache policies
2. **Async Copy & Barriers** - mbarrier, expect_tx, arrive, wait patterns
3. **tcgen05** - Tensor Core Gen 5 MMA instructions, block scaling
4. **TMEM** - Tensor Memory allocation, layout, T2R/R2T copies
5. **Scale Factors** - FP8 scale factor layout, permutation for tcgen05
6. **Warp Specialization** - TMA/MMA/epilogue warp roles
7. **Pipeline Patterns** - Double-buffering, multi-stage pipelines
8. **Integration** - Common pitfalls, alignment, debugging

### Advanced Skills (from DeepGEMM & SonicMoE)

1. **TMA Advanced** - Mixed TMA+CPASYNC, cta_group modes
2. **UMMA** - SS/TS modes, ScaleOut, tile shapes
3. **TMEM Advanced** - Capacity, allocation granularity, copy atoms
4. **WGMMA** - 128-thread warp groups, pingpong buffering
5. **Quantization** - FP8/FP4 formats, accumulation precision
6. **Heuristics** - Block size selection, SMEM constraints
7. **MoE Patterns** - Grouped GEMM, expert routing
8. **Cluster** - Multi-SM coordination, TMA multicast

## Difficulty Distribution

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 18 | Basic concepts, definitions |
| Medium | 24 | Implementation patterns, trade-offs |
| Hard | 14 | Advanced optimization, edge cases |

## Evaluation Workflow

### Option 1: Batch Evaluation (Recommended)

```bash
# 1. Export prompts
python3 run_eval.py export -o prompts.json

# 2. Send prompts to AI and collect responses (your code)
# responses.json should be: {"question_id": "A", "question_id2": "B", ...}

# 3. Import and score
python3 run_eval.py import -p prompts.json -r responses.json -m "model-name"
```

### Option 2: Interactive Evaluation

```bash
# Run interactively, entering AI responses one by one
python3 run_eval.py interactive --model gpt-4 --include-advanced
```

### Option 3: Human Annotation TUI

```bash
# For cases that can't be auto-scored
python3 run_eval.py annotate -f results.json
```

## Output Format

Results are saved as JSON with detailed breakdown:

```json
{
  "metadata": {"model": "gpt-4", "timestamp": "..."},
  "responses": {"tma_easy_1": "A", "tma_easy_2": "B", ...},
  "scores": {"tma_easy_1": 1.0, "tma_easy_2": 1.0, ...},
  "summary": {
    "by_skill": {
      "tma": {"correct": 4, "total": 5, "percentage": 80.0},
      ...
    },
    "by_difficulty": {
      "easy": {"correct": 16, "total": 18, "percentage": 88.9},
      ...
    },
    "overall": {"correct": 42, "total": 56, "percentage": 75.0}
  }
}
```

## Code Challenges

In addition to multiple-choice, there are 5 runnable code challenges:

| ID | Skill | Description |
|----|-------|-------------|
| `code_sf_permute` | Scale Factors | Implement tcgen05 scale factor permutation |
| `code_barrier` | Async Copy | Producer-consumer barrier pattern |
| `code_warp_assign` | Warp Specialization | Calculate warp roles from thread ID |
| `code_pipeline_idx` | Pipeline | Pipeline stage index calculation |
| `code_tmem_layout` | TMEM | Calculate TMEM region offsets |

## Files

```
eval/
├── run_eval.py          # Main entry point
├── questions.py         # Core skill questions (34 MC)
├── questions_advanced.py # Advanced questions (22 MC)
├── code_challenges.py   # Runnable code challenges (5)
├── runner.py            # Evaluation runner & scoring
├── tui.py               # Terminal UI for annotation
└── README.md            # This file
```

## Reference Sources

Questions are extracted from:
- `1.py` - `6.py`: Expert NVFP4 GEMM kernels (fastest to slowest)
- `DeepGEMM/`: Blackwell-specific JIT kernel implementations
- `SonicMoE/`: MoE GEMM with WGMMA and grouped patterns
- `cutlass/`: NVIDIA's CUTLASS library SM100 implementations

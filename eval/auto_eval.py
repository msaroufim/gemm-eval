#!/usr/bin/env python3
"""
NVFP4 GEMM Automated Evaluation

This script is designed to be run by Claude Code to evaluate AI knowledge.
It outputs structured results that can be scored automatically.
"""

import json
from datetime import datetime

# Questions grouped by skill - answers NOT included in prompts
EVAL_QUESTIONS = [
    # TMA
    {
        "id": "tma_1", "skill": "tma", "difficulty": "easy",
        "question": "What does TMA stand for and what is its primary purpose in GEMM kernels?",
        "choices": ["A) Tensor Memory Accelerator - hardware unit for async data transfer from global to shared memory",
                   "B) Thread Memory Access - software abstraction for thread-local storage",
                   "C) Tile Matrix Accumulator - hardware unit for matrix multiplication",
                   "D) Transfer Mode Async - compiler optimization for memory prefetching"],
        "correct": "A"
    },
    {
        "id": "tma_2", "skill": "tma", "difficulty": "easy",
        "question": "Which PTX instruction is used for 3D tensor TMA copies on Hopper/Blackwell?",
        "choices": ["A) cp.sync.tensor.3d",
                   "B) cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes",
                   "C) ld.global.async.3d",
                   "D) memcpy.async.tensor"],
        "correct": "B"
    },
    {
        "id": "tma_3", "skill": "tma", "difficulty": "medium",
        "question": "In TMA operations with mbarrier, what is the correct order of barrier operations for the producer?",
        "choices": ["A) arrive → wait → expect_tx",
                   "B) expect_tx → issue TMA → arrive",
                   "C) wait → expect_tx → issue TMA",
                   "D) issue TMA → expect_tx → arrive"],
        "correct": "B"
    },
    # tcgen05
    {
        "id": "tcgen05_1", "skill": "tcgen05", "difficulty": "easy",
        "question": "What does tcgen05 refer to on NVIDIA Blackwell GPUs?",
        "choices": ["A) Tensor Core Generation 5 - the MMA execution unit on SM100",
                   "B) Thread Compute Group 05 - software threading model",
                   "C) Tensor Compression Gen 05 - data compression unit",
                   "D) Tile Cache Gen 05 - L2 cache controller"],
        "correct": "A"
    },
    {
        "id": "tcgen05_2", "skill": "tcgen05", "difficulty": "medium",
        "question": "In NVFP4 GEMM with block_scale.block16, what does the 'block16' mean?",
        "choices": ["A) Process 16 blocks at a time",
                   "B) Every 16 consecutive K-dimension elements share one FP8 scale factor",
                   "C) Use 16-bit accumulation",
                   "D) Tile size is 16x16"],
        "correct": "B"
    },
    {
        "id": "tcgen05_3", "skill": "tcgen05", "difficulty": "medium",
        "question": "Why must ACCUMULATE be set to False for the first MMA iteration?",
        "choices": ["A) Hardware initialization requirement",
                   "B) First iteration must initialize accumulators to zero; True would add to garbage values",
                   "C) Compiler optimization",
                   "D) Memory alignment requirement"],
        "correct": "B"
    },
    # WGMMA/Warp groups
    {
        "id": "wgmma_1", "skill": "wgmma", "difficulty": "easy",
        "question": "How many threads are in a warp group for WGMMA operations on Hopper/Blackwell?",
        "choices": ["A) 32 threads (1 warp)",
                   "B) 64 threads (2 warps)",
                   "C) 128 threads (4 warps)",
                   "D) 256 threads (8 warps)"],
        "correct": "C"
    },
    {
        "id": "wgmma_2", "skill": "wgmma", "difficulty": "easy",
        "question": "What FP8 formats are supported on NVIDIA Blackwell?",
        "choices": ["A) Only E5M2",
                   "B) E4M3 and E5M2",
                   "C) E3M4 and E4M3",
                   "D) Only E4M3"],
        "correct": "B"
    },
    {
        "id": "quant_1", "skill": "quantization", "difficulty": "medium",
        "question": "What accumulation precision is used for FP8/FP4 GEMM to maintain accuracy?",
        "choices": ["A) FP8 accumulation",
                   "B) FP16 accumulation",
                   "C) FP32 accumulation",
                   "D) INT32 accumulation"],
        "correct": "C"
    },
    # TMEM
    {
        "id": "tmem_1", "skill": "tmem", "difficulty": "easy",
        "question": "What is TMEM on Blackwell and what is it used for?",
        "choices": ["A) Thread Memory - local variables per thread",
                   "B) Tensor Memory - on-chip storage for MMA accumulators and scale factors",
                   "C) Texture Memory - read-only cached data",
                   "D) Tile Memory - L1 cache extension"],
        "correct": "B"
    },
    {
        "id": "tmem_2", "skill": "tmem", "difficulty": "medium",
        "question": "What must be called before using TMEM in a Blackwell kernel?",
        "choices": ["A) tmem.init()",
                   "B) tcgen05.alloc() followed by waiting for allocation to complete",
                   "C) cudaMalloc for TMEM",
                   "D) Just use it directly, no allocation needed"],
        "correct": "B"
    },
    # Async/Barriers
    {
        "id": "async_1", "skill": "async_copy", "difficulty": "easy",
        "question": "What is the purpose of mbarrier (memory barrier) in async copy operations?",
        "choices": ["A) To prevent race conditions between CPU and GPU",
                   "B) To synchronize producer and consumer - producer signals completion, consumer waits",
                   "C) To flush GPU caches to main memory",
                   "D) To serialize all memory operations globally"],
        "correct": "B"
    },
    {
        "id": "async_2", "skill": "async_copy", "difficulty": "easy",
        "question": "What does expect_tx do in the async copy pattern?",
        "choices": ["A) Transmits data to the next stage",
                   "B) Sets the expected number of bytes that will arrive at the barrier",
                   "C) Waits for a transmission to complete",
                   "D) Cancels a pending async operation"],
        "correct": "B"
    },
    # Warp Specialization
    {
        "id": "warp_1", "skill": "warp_spec", "difficulty": "medium",
        "question": "In warp-specialized GEMM kernels, what are the typical warp roles?",
        "choices": ["A) All warps do the same work",
                   "B) TMA warp (loads data), MMA warp (compute), Epilogue warps (store results)",
                   "C) Half compute, half memory",
                   "D) One warp per matrix"],
        "correct": "B"
    },
    {
        "id": "warp_2", "skill": "warp_spec", "difficulty": "hard",
        "question": "In expert NVFP4 kernels with 6 warps (192 threads), how are roles typically assigned?",
        "choices": ["A) All 6 warps do MMA",
                   "B) 4 warps epilogue, 1 warp MMA, 1 warp TMA",
                   "C) 3 warps TMA, 3 warps MMA",
                   "D) 2 warps each for TMA, MMA, epilogue"],
        "correct": "B"
    },
    # Pipeline
    {
        "id": "pipe_1", "skill": "pipeline", "difficulty": "easy",
        "question": "What is the purpose of double-buffering in GEMM kernels?",
        "choices": ["A) Use two GPUs",
                   "B) Overlap data loading with computation - load next tile while computing current",
                   "C) Double the precision",
                   "D) Handle two matrices at once"],
        "correct": "B"
    },
]


def format_prompt_batch(questions, start_idx=0):
    """Format questions into a prompt for Claude."""
    prompt = "Answer these GPU programming questions. Reply with ONLY the letter for each.\n\n"

    for i, q in enumerate(questions):
        prompt += f"Q{start_idx + i + 1} ({q['id']}): {q['question']}\n"
        for choice in q['choices']:
            prompt += f"  {choice}\n"
        prompt += "\n"

    prompt += f"Format your answer as: Q1: X, Q2: X, ... (just letters A-D)"
    return prompt


def generate_answer_key():
    """Generate the answer key."""
    return {q["id"]: q["correct"] for q in EVAL_QUESTIONS}


def score_responses(responses: dict, answer_key: dict) -> dict:
    """Score responses against answer key."""
    results = {
        "correct": 0,
        "total": len(answer_key),
        "by_skill": {},
        "details": []
    }

    for qid, correct in answer_key.items():
        got = responses.get(qid, "?").upper()
        is_correct = got == correct

        if is_correct:
            results["correct"] += 1

        # Find question details
        q = next((q for q in EVAL_QUESTIONS if q["id"] == qid), None)
        skill = q["skill"] if q else "unknown"

        if skill not in results["by_skill"]:
            results["by_skill"][skill] = {"correct": 0, "total": 0}
        results["by_skill"][skill]["total"] += 1
        if is_correct:
            results["by_skill"][skill]["correct"] += 1

        results["details"].append({
            "id": qid,
            "skill": skill,
            "expected": correct,
            "got": got,
            "correct": is_correct
        })

    results["percentage"] = round(results["correct"] / results["total"] * 100, 1)
    return results


def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "=" * 50)
    print("NVFP4 GEMM EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nOverall: {results['correct']}/{results['total']} ({results['percentage']}%)")

    print("\nBy Skill:")
    for skill, stats in sorted(results["by_skill"].items()):
        pct = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        status = "✓" if pct == 100 else "○" if pct >= 50 else "✗"
        print(f"  {status} {skill:15s} {stats['correct']}/{stats['total']} ({pct}%)")

    # Show failures
    failures = [d for d in results["details"] if not d["correct"]]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  ✗ {f['id']}: got {f['got']}, expected {f['expected']}")

    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", action="store_true", help="Print prompt for Claude")
    parser.add_argument("--key", "-k", action="store_true", help="Print answer key")
    parser.add_argument("--score", "-s", help="Score responses from JSON file")

    args = parser.parse_args()

    if args.prompt:
        print(format_prompt_batch(EVAL_QUESTIONS))
    elif args.key:
        key = generate_answer_key()
        print(json.dumps(key, indent=2))
    elif args.score:
        with open(args.score) as f:
            responses = json.load(f)
        key = generate_answer_key()
        results = score_responses(responses, key)
        print_results(results)
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        parser.print_help()
        print("\nQuick test of questions:")
        print(f"Total questions: {len(EVAL_QUESTIONS)}")
        skills = set(q["skill"] for q in EVAL_QUESTIONS)
        print(f"Skills covered: {', '.join(sorted(skills))}")

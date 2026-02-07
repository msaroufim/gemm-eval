#!/usr/bin/env python3
"""
NVFP4 GEMM Evaluation - Self-Test Script

This script generates prompts that can be directly evaluated by Claude.
Run: python3 self_eval.py > prompts.txt
Then paste prompts.txt into a Claude conversation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from questions import get_all_questions
from questions_advanced import get_advanced_questions


def generate_single_prompt_eval(questions: list, batch_size: int = 10):
    """Generate a single prompt with multiple questions for batch evaluation."""

    prompt = """# NVFP4 GEMM Knowledge Evaluation

Answer each question with ONLY the letter (A, B, C, or D).
Format your response as a JSON object: {"question_id": "X", ...}

"""
    for i, q in enumerate(questions[:batch_size]):
        prompt += f"""
---
**Q{i+1}. {q['id']}** [{q['difficulty']}]
{q['question']}

{chr(10).join(q['choices'])}

"""

    prompt += """
---
Reply with JSON only: {"q1_id": "A", "q2_id": "B", ...}
"""
    return prompt


def generate_individual_prompts(questions: list):
    """Generate individual prompts for each question."""
    prompts = []
    for q in questions:
        prompt = f"""NVFP4 GEMM Question [{q['skill']}] [{q['difficulty']}]:

{q['question']}

{chr(10).join(q['choices'])}

Reply with ONLY the letter (A, B, C, or D)."""
        prompts.append({
            "id": q["id"],
            "prompt": prompt,
            "correct": q["correct"],
            "explanation": q["explanation"]
        })
    return prompts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", choices=["batch", "individual", "json"],
                        default="individual", help="Output format")
    parser.add_argument("--skill", "-s", help="Filter by skill")
    parser.add_argument("--difficulty", "-d", help="Filter by difficulty")
    parser.add_argument("--limit", "-n", type=int, help="Limit questions")
    parser.add_argument("--advanced", "-a", action="store_true", help="Include advanced")

    args = parser.parse_args()

    questions = get_all_questions()
    if args.advanced:
        questions += get_advanced_questions()

    if args.skill:
        questions = [q for q in questions if q["skill"] == args.skill]
    if args.difficulty:
        questions = [q for q in questions if q["difficulty"] == args.difficulty]
    if args.limit:
        questions = questions[:args.limit]

    if args.format == "batch":
        print(generate_single_prompt_eval(questions))
    elif args.format == "individual":
        prompts = generate_individual_prompts(questions)
        for p in prompts:
            print(f"\n{'='*60}")
            print(f"ID: {p['id']}")
            print(f"{'='*60}")
            print(p['prompt'])
            print(f"\n[Correct: {p['correct']}]")
    elif args.format == "json":
        prompts = generate_individual_prompts(questions)
        print(json.dumps(prompts, indent=2))

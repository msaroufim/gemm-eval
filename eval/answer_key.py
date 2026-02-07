#!/usr/bin/env python3
"""
NVFP4 GEMM Evaluation - Answer Key Generator

Generates expected answers and explanations for manual comparison.
Run this, then use the TUI to score AI responses against the key.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from questions import get_all_questions
from questions_advanced import get_advanced_questions


def generate_answer_key(include_advanced: bool = True, output: str = "answer_key.json"):
    """Generate complete answer key with explanations."""
    questions = get_all_questions()
    if include_advanced:
        questions += get_advanced_questions()

    answer_key = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_questions": len(questions)
        },
        "questions": []
    }

    for q in questions:
        answer_key["questions"].append({
            "id": q["id"],
            "skill": q["skill"],
            "skill_name": q["skill_name"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "choices": q["choices"],
            "correct_answer": q["correct"],
            "explanation": q["explanation"],
            "reference": q.get("reference", "")
        })

    with open(output, 'w') as f:
        json.dump(answer_key, f, indent=2)

    print(f"✓ Generated answer key with {len(questions)} questions")
    print(f"✓ Saved to: {output}")
    return answer_key


def print_quiz(skill: str = None, difficulty: str = None, show_answers: bool = False):
    """Print questions in quiz format."""
    questions = get_all_questions() + get_advanced_questions()

    if skill:
        questions = [q for q in questions if q["skill"] == skill]
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]

    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Q{i+1}. [{q['skill']}] [{q['difficulty'].upper()}]")
        print(f"{'='*60}")
        print(f"\n{q['question']}\n")
        for choice in q['choices']:
            print(f"  {choice}")

        if show_answers:
            print(f"\n  → Answer: {q['correct']}")
            print(f"  → {q['explanation']}")
        else:
            print("\n  Your answer: ___")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", "-k", action="store_true", help="Generate answer key JSON")
    parser.add_argument("--quiz", "-q", action="store_true", help="Print quiz format")
    parser.add_argument("--skill", "-s", help="Filter by skill")
    parser.add_argument("--difficulty", "-d", help="Filter by difficulty")
    parser.add_argument("--answers", "-a", action="store_true", help="Show answers in quiz")
    parser.add_argument("--output", "-o", default="answer_key.json", help="Output file")

    args = parser.parse_args()

    if args.key:
        generate_answer_key(output=args.output)
    elif args.quiz:
        print_quiz(skill=args.skill, difficulty=args.difficulty, show_answers=args.answers)
    else:
        parser.print_help()

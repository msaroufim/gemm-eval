#!/usr/bin/env python3
"""
Automated NVFP4 GEMM Evaluation using Claude

Spawns Claude processes to answer each question and scores results.
No GPU required - tests knowledge only.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from questions import QUESTIONS, get_all_questions
from questions_advanced import ADVANCED_QUESTIONS, get_advanced_questions


class ClaudeEvaluator:
    """Runs evaluation by spawning Claude processes."""

    def __init__(self, model: str = "sonnet", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "version": "1.0"
            },
            "responses": {},
            "scores": {},
            "details": []
        }

    def ask_claude(self, prompt: str) -> str:
        """Spawn Claude CLI to answer a question."""
        try:
            # Use claude CLI with --print flag for non-interactive output
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", self.model],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except FileNotFoundError:
            print("Error: 'claude' CLI not found. Make sure Claude Code is installed.")
            sys.exit(1)
        except Exception as e:
            return f"[ERROR: {e}]"

    def format_mc_prompt(self, question: dict) -> str:
        """Format a multiple-choice question as a prompt."""
        return f"""You are being evaluated on NVFP4 GEMM implementation for NVIDIA Blackwell GPUs.

Question: {question['question']}

Choices:
{chr(10).join(question['choices'])}

Reply with ONLY the letter (A, B, C, or D). Nothing else."""

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer letter from Claude's response."""
        response = response.strip().upper()

        # Direct single letter
        if response in ['A', 'B', 'C', 'D']:
            return response

        # Starts with letter
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return response[0]

        # Look for pattern like "A)" or "A."
        match = re.search(r'\b([ABCD])[).\s:]', response)
        if match:
            return match.group(1)

        # Look for "answer is X" pattern
        match = re.search(r'answer\s+is\s+([ABCD])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def score_answer(self, extracted: Optional[str], correct: str) -> float:
        """Score an answer."""
        if extracted is None:
            return 0.0
        return 1.0 if extracted == correct.upper() else 0.0

    def run_question(self, question: dict, idx: int, total: int) -> dict:
        """Run a single question and return result."""
        qid = question["id"]
        correct = question["correct"]

        # Format prompt
        prompt = self.format_mc_prompt(question)

        # Get response from Claude
        if self.verbose:
            print(f"  Asking Claude...")

        response = self.ask_claude(prompt)

        # Extract and score
        extracted = self.extract_answer(response)
        score = self.score_answer(extracted, correct)

        # Store results
        self.results["responses"][qid] = response
        self.results["scores"][qid] = score

        result = {
            "id": qid,
            "skill": question["skill"],
            "difficulty": question["difficulty"],
            "correct_answer": correct,
            "claude_response": response,
            "extracted_answer": extracted,
            "score": score,
            "passed": score == 1.0
        }
        self.results["details"].append(result)

        # Print status
        status = "✓" if score == 1.0 else "✗"
        extracted_str = extracted or "?"
        print(f"[{idx+1}/{total}] {qid:25s} {status} (got: {extracted_str}, expected: {correct})")

        if self.verbose and score == 0.0:
            print(f"      Response: {response[:80]}...")

        return result

    def run_evaluation(self, questions: list, skill_filter: Optional[str] = None,
                       difficulty_filter: Optional[str] = None, limit: Optional[int] = None):
        """Run full evaluation on questions."""
        # Apply filters
        if skill_filter:
            questions = [q for q in questions if q["skill"] == skill_filter]
        if difficulty_filter:
            questions = [q for q in questions if q["difficulty"] == difficulty_filter]
        if limit:
            questions = questions[:limit]

        if not questions:
            print("No questions match the filters.")
            return

        print(f"\n{'='*60}")
        print(f"NVFP4 GEMM EVALUATION - {len(questions)} questions")
        print(f"Model: {self.model}")
        print(f"{'='*60}\n")

        for i, q in enumerate(questions):
            self.run_question(q, i, len(questions))

        self.print_summary()
        return self.results

    def print_summary(self):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        # By skill
        from collections import defaultdict
        skill_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for detail in self.results["details"]:
            skill = detail["skill"]
            diff = detail["difficulty"]
            score = detail["score"]

            skill_stats[skill]["total"] += 1
            skill_stats[skill]["correct"] += score
            diff_stats[diff]["total"] += 1
            diff_stats[diff]["correct"] += score

        print("\nBy Skill:")
        print("-" * 40)
        for skill, stats in sorted(skill_stats.items()):
            pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            bar = self._make_bar(pct)
            print(f"  {skill:20s} {stats['correct']:4.0f}/{stats['total']:<3d} {bar} {pct:5.1f}%")

        print("\nBy Difficulty:")
        print("-" * 40)
        for diff in ["easy", "medium", "hard"]:
            if diff in diff_stats:
                stats = diff_stats[diff]
                pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                bar = self._make_bar(pct)
                print(f"  {diff:20s} {stats['correct']:4.0f}/{stats['total']:<3d} {bar} {pct:5.1f}%")

        # Overall
        total_correct = sum(s["correct"] for s in skill_stats.values())
        total_questions = sum(s["total"] for s in skill_stats.values())
        overall_pct = (total_correct / total_questions * 100) if total_questions > 0 else 0

        print("-" * 40)
        bar = self._make_bar(overall_pct)
        print(f"  {'OVERALL':20s} {total_correct:4.0f}/{total_questions:<3d} {bar} {overall_pct:5.1f}%")
        print(f"{'='*60}\n")

        # Print failures
        failures = [d for d in self.results["details"] if not d["passed"]]
        if failures:
            print("FAILURES:")
            print("-" * 40)
            for f in failures:
                print(f"  ✗ {f['id']} - got {f['extracted_answer'] or '?'}, expected {f['correct_answer']}")

    def _make_bar(self, pct: float, width: int = 15) -> str:
        filled = int(width * pct / 100)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        # Compute summary
        from collections import defaultdict
        skill_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for detail in self.results["details"]:
            skill_stats[detail["skill"]]["total"] += 1
            skill_stats[detail["skill"]]["correct"] += detail["score"]

        self.results["summary"] = {
            "by_skill": {k: {**v, "percentage": (v["correct"]/v["total"]*100) if v["total"] > 0 else 0}
                        for k, v in skill_stats.items()},
            "overall": {
                "correct": sum(v["correct"] for v in skill_stats.values()),
                "total": sum(v["total"] for v in skill_stats.values())
            }
        }
        self.results["summary"]["overall"]["percentage"] = (
            self.results["summary"]["overall"]["correct"] /
            self.results["summary"]["overall"]["total"] * 100
        ) if self.results["summary"]["overall"]["total"] > 0 else 0

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📁 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run NVFP4 GEMM eval using Claude")
    parser.add_argument("--model", "-m", default="sonnet",
                        choices=["sonnet", "opus", "haiku"],
                        help="Claude model to use")
    parser.add_argument("--skill", "-s", help="Filter by skill ID")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--limit", "-n", type=int, help="Limit number of questions")
    parser.add_argument("--include-advanced", "-a", action="store_true",
                        help="Include advanced questions from DeepGEMM/SonicMoE")
    parser.add_argument("--output", "-o", help="Output file for results",
                        default=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed responses")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available skills and exit")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable skills:")
        print("-" * 40)
        print("\nCore:")
        for skill_id, data in QUESTIONS.items():
            print(f"  {skill_id:20s} - {data['name']} ({len(data['questions'])} q)")
        print("\nAdvanced:")
        for skill_id, data in ADVANCED_QUESTIONS.items():
            print(f"  {skill_id:20s} - {data['name']} ({len(data['questions'])} q)")
        return

    # Collect questions
    questions = get_all_questions()
    if args.include_advanced:
        questions += get_advanced_questions()

    # Run evaluation
    evaluator = ClaudeEvaluator(model=args.model, verbose=args.verbose)
    evaluator.run_evaluation(
        questions,
        skill_filter=args.skill,
        difficulty_filter=args.difficulty,
        limit=args.limit
    )
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()

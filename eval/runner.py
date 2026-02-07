"""
NVFP4 GEMM Evaluation Runner

Run evaluations on AI models and collect/score responses.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from questions import QUESTIONS, get_all_questions, get_questions_by_skill, get_skill_list


class EvalRunner:
    """Run evaluations and collect responses."""

    def __init__(self, output_dir: str = "eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": None,
                "version": "1.0"
            },
            "responses": {},
            "scores": {},
            "summary": {}
        }

    def set_model(self, model_name: str):
        """Set the model being evaluated."""
        self.results["metadata"]["model"] = model_name

    def format_question_prompt(self, question: dict) -> str:
        """Format a question as a prompt for the AI."""
        prompt = f"""You are being evaluated on your knowledge of NVFP4 GEMM implementation on NVIDIA Blackwell GPUs.

**Question ({question['difficulty'].upper()}):**
{question['question']}

**Choices:**
{chr(10).join(question['choices'])}

**Instructions:**
- Reply with ONLY the letter of your answer (A, B, C, or D)
- Do not explain your reasoning
- Just the letter

Your answer:"""
        return prompt

    def format_question_with_explanation(self, question: dict) -> str:
        """Format a question that asks for explanation (for harder eval)."""
        prompt = f"""You are being evaluated on your knowledge of NVFP4 GEMM implementation on NVIDIA Blackwell GPUs.

**Question ({question['difficulty'].upper()}):**
{question['question']}

**Choices:**
{chr(10).join(question['choices'])}

**Instructions:**
- First, state your answer letter (A, B, C, or D)
- Then briefly explain why (1-2 sentences)

Your answer:"""
        return prompt

    def auto_score(self, question: dict, response: str) -> Optional[float]:
        """
        Automatically score a multiple-choice response.

        Returns:
            1.0 for correct, 0.0 for incorrect, None if can't determine
        """
        if not response:
            return None

        # Extract the letter from response
        response_clean = response.strip().upper()

        # Try to find a single letter answer
        correct = question["correct"].upper()

        # Direct match
        if response_clean == correct:
            return 1.0

        # Starts with the letter
        if response_clean.startswith(correct + ")") or response_clean.startswith(correct + "."):
            return 1.0
        if response_clean.startswith(correct + " "):
            return 1.0
        if response_clean[0:1] == correct:
            # Check if it's clearly this answer
            return 1.0

        # Check for wrong answers
        wrong_letters = [l for l in "ABCD" if l != correct]
        for wrong in wrong_letters:
            if response_clean == wrong or response_clean.startswith(wrong + ")") or response_clean.startswith(wrong + " "):
                return 0.0
            if response_clean[0:1] == wrong:
                return 0.0

        # Can't determine
        return None

    def record_response(self, question_id: str, response: str, score: Optional[float] = None):
        """Record an AI response and optional score."""
        self.results["responses"][question_id] = response
        if score is not None:
            self.results["scores"][question_id] = score

    def compute_summary(self):
        """Compute summary statistics."""
        from collections import defaultdict

        questions = get_all_questions()
        skill_stats = defaultdict(lambda: {"correct": 0, "total": 0, "questions": []})

        for q in questions:
            qid = q["id"]
            skill = q["skill"]

            if qid in self.results["scores"]:
                score = self.results["scores"][qid]
                skill_stats[skill]["total"] += 1
                skill_stats[skill]["correct"] += score
                skill_stats[skill]["questions"].append({
                    "id": qid,
                    "score": score,
                    "difficulty": q["difficulty"]
                })

        self.results["summary"] = {
            "by_skill": {},
            "by_difficulty": {"easy": {"correct": 0, "total": 0},
                             "medium": {"correct": 0, "total": 0},
                             "hard": {"correct": 0, "total": 0}},
            "overall": {"correct": 0, "total": 0}
        }

        for skill, stats in skill_stats.items():
            pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            self.results["summary"]["by_skill"][skill] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "percentage": round(pct, 1),
                "questions": stats["questions"]
            }
            self.results["summary"]["overall"]["correct"] += stats["correct"]
            self.results["summary"]["overall"]["total"] += stats["total"]

            # By difficulty
            for q_info in stats["questions"]:
                diff = q_info["difficulty"]
                self.results["summary"]["by_difficulty"][diff]["total"] += 1
                self.results["summary"]["by_difficulty"][diff]["correct"] += q_info["score"]

        # Overall percentage
        overall = self.results["summary"]["overall"]
        overall["percentage"] = round(
            (overall["correct"] / overall["total"] * 100) if overall["total"] > 0 else 0, 1
        )

        # Difficulty percentages
        for diff, stats in self.results["summary"]["by_difficulty"].items():
            stats["percentage"] = round(
                (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0, 1
            )

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        self.compute_summary()

        if filename is None:
            model_name = self.results["metadata"]["model"] or "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{model_name}_{timestamp}.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        return filepath

    def print_report(self):
        """Print a formatted report to console."""
        self.compute_summary()

        print("\n" + "="*60)
        print("NVFP4 GEMM EVALUATION REPORT".center(60))
        print("="*60)

        meta = self.results["metadata"]
        print(f"\nModel: {meta.get('model', 'Unknown')}")
        print(f"Date:  {meta.get('timestamp', 'Unknown')}")

        print("\n" + "-"*40)
        print("SCORES BY SKILL")
        print("-"*40)

        summary = self.results["summary"]
        for skill_id, stats in sorted(summary["by_skill"].items()):
            bar = self._make_bar(stats["percentage"])
            print(f"{skill_id:20s} {stats['correct']:4.1f}/{stats['total']:<3d} {bar} {stats['percentage']:5.1f}%")

        print("\n" + "-"*40)
        print("SCORES BY DIFFICULTY")
        print("-"*40)

        for diff in ["easy", "medium", "hard"]:
            stats = summary["by_difficulty"][diff]
            bar = self._make_bar(stats["percentage"])
            print(f"{diff.upper():20s} {stats['correct']:4.1f}/{stats['total']:<3d} {bar} {stats['percentage']:5.1f}%")

        print("\n" + "-"*40)
        overall = summary["overall"]
        bar = self._make_bar(overall["percentage"])
        print(f"{'OVERALL':20s} {overall['correct']:4.1f}/{overall['total']:<3d} {bar} {overall['percentage']:5.1f}%")
        print("="*60 + "\n")

    def _make_bar(self, percentage: float, width: int = 20) -> str:
        """Make a simple ASCII progress bar."""
        filled = int(width * percentage / 100)
        empty = width - filled
        return f"[{'█'*filled}{'░'*empty}]"

    def get_failures(self) -> List[dict]:
        """Get list of failed questions with details."""
        questions = get_all_questions()
        failures = []

        for q in questions:
            qid = q["id"]
            if qid in self.results["scores"] and self.results["scores"][qid] == 0:
                failures.append({
                    "id": qid,
                    "skill": q["skill"],
                    "difficulty": q["difficulty"],
                    "question": q["question"],
                    "correct_answer": q["correct"],
                    "explanation": q["explanation"],
                    "ai_response": self.results["responses"].get(qid, ""),
                    "reference": q.get("reference", "")
                })

        return failures

    def print_failures(self):
        """Print detailed failure analysis."""
        failures = self.get_failures()

        if not failures:
            print("\n✓ No failures!")
            return

        print("\n" + "="*60)
        print("FAILURE ANALYSIS".center(60))
        print("="*60)

        for f in failures:
            print(f"\n✗ {f['id']} ({f['skill']}, {f['difficulty']})")
            print(f"  Q: {f['question'][:70]}...")
            print(f"  Expected: {f['correct_answer']}")
            print(f"  Got: {f['ai_response'][:50]}..." if len(f['ai_response']) > 50 else f"  Got: {f['ai_response']}")
            print(f"  Why: {f['explanation'][:80]}...")


def run_interactive_eval(runner: EvalRunner, skill: Optional[str] = None):
    """Run evaluation interactively (manual mode)."""
    if skill:
        questions = get_questions_by_skill(skill)
        if not questions:
            print(f"Unknown skill: {skill}")
            print("Available skills:", [s[0] for s in get_skill_list()])
            return
    else:
        questions = get_all_questions()

    print(f"\nRunning evaluation on {len(questions)} questions...")
    print("For each question, enter the AI's response (or 'q' to quit)\n")

    for i, q in enumerate(questions):
        print(f"\n{'='*50}")
        print(f"Question {i+1}/{len(questions)} [{q['skill']}] [{q['difficulty']}]")
        print(f"{'='*50}")
        print(f"\n{q['question']}\n")
        for choice in q['choices']:
            print(f"  {choice}")
        print()

        response = input("AI Response (letter or 'q' to quit): ").strip()
        if response.lower() == 'q':
            break

        score = runner.auto_score(q, response)
        runner.record_response(q["id"], response, score)

        if score == 1.0:
            print("✓ Correct!")
        elif score == 0.0:
            print(f"✗ Wrong. Correct answer: {q['correct']}")
            print(f"  Explanation: {q['explanation']}")
        else:
            print("? Could not auto-score. Will need manual review.")


def export_prompts(output_file: str, skill: Optional[str] = None, with_explanation: bool = False):
    """Export all questions as prompts for batch evaluation."""
    runner = EvalRunner()

    if skill:
        questions = get_questions_by_skill(skill)
    else:
        questions = get_all_questions()

    prompts = []
    for q in questions:
        if with_explanation:
            prompt = runner.format_question_with_explanation(q)
        else:
            prompt = runner.format_question_prompt(q)

        prompts.append({
            "id": q["id"],
            "skill": q["skill"],
            "difficulty": q["difficulty"],
            "prompt": prompt,
            "correct_answer": q["correct"]
        })

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Exported {len(prompts)} prompts to {output_file}")


def import_responses(prompts_file: str, responses_file: str, output_dir: str = "eval_results"):
    """Import AI responses and score them."""
    runner = EvalRunner(output_dir)

    with open(prompts_file) as f:
        prompts = json.load(f)

    with open(responses_file) as f:
        responses = json.load(f)

    # Build question lookup
    questions = {q["id"]: q for q in get_all_questions()}

    for prompt in prompts:
        qid = prompt["id"]
        if qid in responses:
            response = responses[qid]
            q = questions[qid]
            score = runner.auto_score(q, response)
            runner.record_response(qid, response, score)

    runner.print_report()
    result_file = runner.save_results()
    print(f"\nResults saved to: {result_file}")

    runner.print_failures()


def main():
    parser = argparse.ArgumentParser(description="NVFP4 GEMM AI Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Interactive eval
    interactive = subparsers.add_parser("interactive", help="Run interactive evaluation")
    interactive.add_argument("--skill", "-s", help="Evaluate specific skill only")
    interactive.add_argument("--model", "-m", help="Model name for metadata")

    # Export prompts
    export = subparsers.add_parser("export", help="Export prompts for batch evaluation")
    export.add_argument("--output", "-o", required=True, help="Output JSON file")
    export.add_argument("--skill", "-s", help="Export specific skill only")
    export.add_argument("--explain", "-e", action="store_true", help="Include explanation request")

    # Import responses
    import_cmd = subparsers.add_parser("import", help="Import and score responses")
    import_cmd.add_argument("--prompts", "-p", required=True, help="Prompts JSON file")
    import_cmd.add_argument("--responses", "-r", required=True, help="Responses JSON file")

    # List skills
    list_cmd = subparsers.add_parser("list", help="List available skills")

    # Show report
    report = subparsers.add_parser("report", help="Show report from results file")
    report.add_argument("--file", "-f", required=True, help="Results JSON file")

    # TUI annotation
    annotate = subparsers.add_parser("annotate", help="Manual annotation TUI")
    annotate.add_argument("--file", "-f", help="Results file to annotate")

    args = parser.parse_args()

    if args.command == "interactive":
        runner = EvalRunner()
        if args.model:
            runner.set_model(args.model)
        run_interactive_eval(runner, args.skill)
        runner.print_report()
        result_file = runner.save_results()
        print(f"\nResults saved to: {result_file}")

    elif args.command == "export":
        export_prompts(args.output, args.skill, args.explain)

    elif args.command == "import":
        import_responses(args.prompts, args.responses)

    elif args.command == "list":
        print("\nAvailable Skills:")
        print("-"*40)
        for skill_id, skill_name in get_skill_list():
            q_count = len(get_questions_by_skill(skill_id))
            print(f"  {skill_id:20s} ({q_count} questions) - {skill_name}")

    elif args.command == "report":
        with open(args.file) as f:
            results = json.load(f)
        runner = EvalRunner()
        runner.results = results
        runner.print_report()
        runner.print_failures()

    elif args.command == "annotate":
        from tui import run_annotation_tui
        run_annotation_tui(args.file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

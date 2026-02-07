#!/usr/bin/env python3
"""
NVFP4 GEMM AI Evaluation Framework

Main entry point for running evaluations on AI models.

Usage:
    # List all skills and questions
    python run_eval.py list

    # Export prompts for batch evaluation
    python run_eval.py export -o prompts.json

    # Run interactive evaluation
    python run_eval.py interactive --model gpt-4

    # Import responses and score
    python run_eval.py import -p prompts.json -r responses.json

    # Run TUI for manual annotation
    python run_eval.py annotate -f results.json

    # Show report from results
    python run_eval.py report -f results.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from questions import QUESTIONS, get_all_questions, get_skill_list
from questions_advanced import ADVANCED_QUESTIONS, get_advanced_questions
from runner import EvalRunner, run_interactive_eval, export_prompts
from code_challenges import CODE_CHALLENGES, get_code_challenges


def list_all(args):
    """List all available skills and question counts."""
    print("\n" + "="*60)
    print("NVFP4 GEMM EVALUATION - AVAILABLE CONTENT")
    print("="*60)

    print("\n📚 CORE SKILLS (from 1.py-6.py expert kernels)")
    print("-"*50)
    total_core = 0
    for skill_id, skill_data in QUESTIONS.items():
        count = len(skill_data["questions"])
        total_core += count
        easy = sum(1 for q in skill_data["questions"] if q["difficulty"] == "easy")
        med = sum(1 for q in skill_data["questions"] if q["difficulty"] == "medium")
        hard = sum(1 for q in skill_data["questions"] if q["difficulty"] == "hard")
        print(f"  {skill_id:20s} {count:2d} questions  (E:{easy} M:{med} H:{hard})")
        print(f"    └─ {skill_data['name']}")

    print(f"\n  Total core: {total_core} questions")

    print("\n🚀 ADVANCED SKILLS (from DeepGEMM & SonicMoE)")
    print("-"*50)
    total_adv = 0
    for skill_id, skill_data in ADVANCED_QUESTIONS.items():
        count = len(skill_data["questions"])
        total_adv += count
        easy = sum(1 for q in skill_data["questions"] if q["difficulty"] == "easy")
        med = sum(1 for q in skill_data["questions"] if q["difficulty"] == "medium")
        hard = sum(1 for q in skill_data["questions"] if q["difficulty"] == "hard")
        print(f"  {skill_id:20s} {count:2d} questions  (E:{easy} M:{med} H:{hard})")
        print(f"    └─ {skill_data['name']}")

    print(f"\n  Total advanced: {total_adv} questions")

    print("\n💻 CODE CHALLENGES (runnable)")
    print("-"*50)
    for cid, challenge in CODE_CHALLENGES.items():
        print(f"  {challenge['id']:20s} [{challenge['difficulty']}]")
        print(f"    └─ {challenge['description']}")

    print(f"\n  Total code challenges: {len(CODE_CHALLENGES)}")

    print("\n" + "="*60)
    print(f"GRAND TOTAL: {total_core + total_adv} MC questions + {len(CODE_CHALLENGES)} code challenges")
    print("="*60 + "\n")


def export_all(args):
    """Export all prompts to JSON file."""
    # Get all questions
    core_questions = get_all_questions()
    adv_questions = get_advanced_questions()
    code_challenges = get_code_challenges()

    all_content = {
        "metadata": {
            "version": "1.0",
            "generated": datetime.now().isoformat(),
            "total_mc_questions": len(core_questions) + len(adv_questions),
            "total_code_challenges": len(code_challenges)
        },
        "multiple_choice": [],
        "code_challenges": []
    }

    runner = EvalRunner()

    # Export MC questions
    for q in core_questions + adv_questions:
        if args.explain:
            prompt = runner.format_question_with_explanation(q)
        else:
            prompt = runner.format_question_prompt(q)

        all_content["multiple_choice"].append({
            "id": q["id"],
            "skill": q["skill"],
            "skill_name": q["skill_name"],
            "difficulty": q["difficulty"],
            "prompt": prompt,
            "correct_answer": q["correct"],
            "explanation": q["explanation"],
            "reference": q.get("reference", "")
        })

    # Export code challenges
    for c in code_challenges:
        all_content["code_challenges"].append({
            "id": c["id"],
            "skill": c["skill"],
            "difficulty": c["difficulty"],
            "description": c["description"],
            "prompt": c["prompt"],
            "solution": c["solution"]
        })

    with open(args.output, 'w') as f:
        json.dump(all_content, f, indent=2)

    print(f"✓ Exported {len(all_content['multiple_choice'])} MC questions")
    print(f"✓ Exported {len(all_content['code_challenges'])} code challenges")
    print(f"✓ Saved to: {args.output}")


def run_interactive(args):
    """Run interactive evaluation session."""
    # Combine all questions
    all_questions = get_all_questions()
    if args.include_advanced:
        all_questions += get_advanced_questions()

    if args.skill:
        all_questions = [q for q in all_questions if q["skill"] == args.skill]

    if args.difficulty:
        all_questions = [q for q in all_questions if q["difficulty"] == args.difficulty]

    if not all_questions:
        print("No questions match the filters.")
        return

    runner = EvalRunner()
    if args.model:
        runner.set_model(args.model)

    print(f"\n🎯 Starting evaluation with {len(all_questions)} questions")
    print("Enter the AI's answer for each question (or 'q' to quit)\n")

    for i, q in enumerate(all_questions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(all_questions)}] {q['skill']} | {q['difficulty'].upper()}")
        print(f"{'='*60}")
        print(f"\n{q['question']}\n")

        for choice in q['choices']:
            print(f"  {choice}")

        print()
        response = input("AI's answer: ").strip()

        if response.lower() == 'q':
            break

        score = runner.auto_score(q, response)
        runner.record_response(q["id"], response, score)

        if score == 1.0:
            print("✓ Correct!")
        elif score == 0.0:
            print(f"✗ Wrong. Correct: {q['correct']}")
            print(f"  → {q['explanation'][:100]}...")
        else:
            print("? Could not auto-score")

    runner.print_report()
    result_file = runner.save_results()
    print(f"\n📁 Results saved to: {result_file}")
    runner.print_failures()


def run_annotate(args):
    """Run TUI for manual annotation."""
    from tui import run_annotation_tui
    run_annotation_tui(args.file)


def show_report(args):
    """Show report from results file."""
    with open(args.file) as f:
        results = json.load(f)

    runner = EvalRunner()
    runner.results = results
    runner.print_report()
    runner.print_failures()


def import_responses(args):
    """Import responses from JSON and score."""
    with open(args.prompts) as f:
        prompts_data = json.load(f)

    with open(args.responses) as f:
        responses = json.load(f)

    # Build question lookup
    all_questions = get_all_questions() + get_advanced_questions()
    questions = {q["id"]: q for q in all_questions}

    runner = EvalRunner()
    if args.model:
        runner.set_model(args.model)

    # Score MC questions
    if "multiple_choice" in prompts_data:
        for prompt in prompts_data["multiple_choice"]:
            qid = prompt["id"]
            if qid in responses and qid in questions:
                response = responses[qid]
                score = runner.auto_score(questions[qid], response)
                runner.record_response(qid, response, score)
    else:
        # Flat format
        for prompt in prompts_data:
            qid = prompt["id"]
            if qid in responses and qid in questions:
                response = responses[qid]
                score = runner.auto_score(questions[qid], response)
                runner.record_response(qid, response, score)

    runner.print_report()
    result_file = runner.save_results()
    print(f"\n📁 Results saved to: {result_file}")
    runner.print_failures()


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 GEMM AI Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List
    list_parser = subparsers.add_parser("list", help="List all skills and questions")

    # Export
    export_parser = subparsers.add_parser("export", help="Export prompts for batch evaluation")
    export_parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    export_parser.add_argument("-e", "--explain", action="store_true",
                               help="Include explanation request in prompts")

    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive evaluation")
    interactive_parser.add_argument("-m", "--model", help="Model name for metadata")
    interactive_parser.add_argument("-s", "--skill", help="Filter by skill ID")
    interactive_parser.add_argument("-d", "--difficulty", choices=["easy", "medium", "hard"],
                                    help="Filter by difficulty")
    interactive_parser.add_argument("--include-advanced", action="store_true",
                                    help="Include advanced questions from DeepGEMM/SonicMoE")

    # Import
    import_parser = subparsers.add_parser("import", help="Import and score responses")
    import_parser.add_argument("-p", "--prompts", required=True, help="Prompts JSON file")
    import_parser.add_argument("-r", "--responses", required=True, help="Responses JSON file")
    import_parser.add_argument("-m", "--model", help="Model name for metadata")

    # Annotate
    annotate_parser = subparsers.add_parser("annotate", help="Manual annotation TUI")
    annotate_parser.add_argument("-f", "--file", help="Results file to annotate")

    # Report
    report_parser = subparsers.add_parser("report", help="Show report from results file")
    report_parser.add_argument("-f", "--file", required=True, help="Results JSON file")

    args = parser.parse_args()

    if args.command == "list":
        list_all(args)
    elif args.command == "export":
        export_all(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "import":
        import_responses(args)
    elif args.command == "annotate":
        run_annotate(args)
    elif args.command == "report":
        show_report(args)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("  python run_eval.py list                    # See all questions")
        print("  python run_eval.py export -o prompts.json  # Export for batch eval")
        print("  python run_eval.py interactive             # Run interactively")


if __name__ == "__main__":
    main()

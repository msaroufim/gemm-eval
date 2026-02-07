"""
Simple TUI for human annotation of AI responses.

Navigate through each question, see AI response, mark correct/incorrect.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Simple terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def clear_screen():
    print('\033[2J\033[H', end='')

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_question(q, idx, total):
    """Display a question with formatting."""
    difficulty_colors = {
        "easy": Colors.GREEN,
        "medium": Colors.YELLOW,
        "hard": Colors.RED
    }
    diff_color = difficulty_colors.get(q["difficulty"], Colors.ENDC)

    print(f"{Colors.DIM}Question {idx+1}/{total}{Colors.ENDC}")
    print(f"{Colors.BOLD}Skill:{Colors.ENDC} {q['skill_name']}")
    print(f"{Colors.BOLD}Difficulty:{Colors.ENDC} {diff_color}{q['difficulty'].upper()}{Colors.ENDC}")
    print(f"{Colors.BOLD}ID:{Colors.ENDC} {q['id']}")
    print()
    print(f"{Colors.CYAN}{q['question']}{Colors.ENDC}")
    print()
    for choice in q["choices"]:
        print(f"  {choice}")
    print()
    print(f"{Colors.DIM}Correct Answer: {q['correct']}{Colors.ENDC}")
    print(f"{Colors.DIM}Reference: {q.get('reference', 'N/A')}{Colors.ENDC}")

def print_ai_response(response):
    """Display AI response."""
    print(f"\n{Colors.BOLD}AI Response:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'─'*50}{Colors.ENDC}")
    print(response if response else "(No response recorded)")
    print(f"{Colors.BLUE}{'─'*50}{Colors.ENDC}")

def get_annotation():
    """Get user annotation for current question."""
    print(f"\n{Colors.BOLD}Your Annotation:{Colors.ENDC}")
    print(f"  [{Colors.GREEN}c{Colors.ENDC}] Correct")
    print(f"  [{Colors.RED}w{Colors.ENDC}] Wrong")
    print(f"  [{Colors.YELLOW}p{Colors.ENDC}] Partial credit (0.5)")
    print(f"  [{Colors.CYAN}s{Colors.ENDC}] Skip (no score)")
    print(f"  [{Colors.DIM}n{Colors.ENDC}] Add note")
    print(f"  [{Colors.DIM}q{Colors.ENDC}] Quit and save")
    print()

    while True:
        choice = input(f"{Colors.BOLD}> {Colors.ENDC}").strip().lower()
        if choice in ['c', 'w', 'p', 's', 'n', 'q']:
            return choice
        print("Invalid choice. Try again.")

def run_annotation_tui(results_file=None, questions=None):
    """
    Run the TUI for annotating AI responses.

    Args:
        results_file: Path to JSON file with AI responses
        questions: List of question dicts (if not using file)
    """
    from questions import get_all_questions

    # Load questions
    all_questions = questions or get_all_questions()

    # Load existing results or create new
    results = {
        "timestamp": datetime.now().isoformat(),
        "annotations": {},
        "notes": {},
        "ai_responses": {}
    }

    if results_file and Path(results_file).exists():
        with open(results_file) as f:
            results = json.load(f)

    # Track position
    current_idx = 0

    # Find first unannotated
    for i, q in enumerate(all_questions):
        if q["id"] not in results["annotations"]:
            current_idx = i
            break

    while current_idx < len(all_questions):
        clear_screen()
        q = all_questions[current_idx]

        print_header("NVFP4 GEMM Eval Annotation")
        print_question(q, current_idx, len(all_questions))

        # Show AI response if available
        ai_resp = results.get("ai_responses", {}).get(q["id"], "")
        print_ai_response(ai_resp)

        # Show existing annotation if any
        if q["id"] in results["annotations"]:
            score = results["annotations"][q["id"]]
            score_str = {1: "CORRECT", 0: "WRONG", 0.5: "PARTIAL"}.get(score, "SKIPPED")
            print(f"\n{Colors.DIM}Current annotation: {score_str}{Colors.ENDC}")

        choice = get_annotation()

        if choice == 'c':
            results["annotations"][q["id"]] = 1
            current_idx += 1
        elif choice == 'w':
            results["annotations"][q["id"]] = 0
            current_idx += 1
        elif choice == 'p':
            results["annotations"][q["id"]] = 0.5
            current_idx += 1
        elif choice == 's':
            results["annotations"][q["id"]] = None
            current_idx += 1
        elif choice == 'n':
            note = input("Enter note: ")
            results["notes"][q["id"]] = note
        elif choice == 'q':
            break

    # Save results
    output_file = results_file or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    clear_screen()
    print_header("Annotation Complete")
    print_summary(results, all_questions)
    print(f"\n{Colors.GREEN}Results saved to: {output_file}{Colors.ENDC}")

    return results

def print_summary(results, all_questions):
    """Print scoring summary."""
    from collections import defaultdict

    annotations = results.get("annotations", {})

    # Group by skill
    skill_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for q in all_questions:
        qid = q["id"]
        skill = q["skill"]

        if qid in annotations and annotations[qid] is not None:
            skill_scores[skill]["total"] += 1
            skill_scores[skill]["correct"] += annotations[qid]

    # Print per-skill
    print(f"\n{Colors.BOLD}Score by Skill:{Colors.ENDC}")
    print(f"{'─'*40}")

    total_correct = 0
    total_questions = 0

    for skill, scores in sorted(skill_scores.items()):
        correct = scores["correct"]
        total = scores["total"]
        pct = (correct / total * 100) if total > 0 else 0

        # Color based on score
        if pct >= 80:
            color = Colors.GREEN
        elif pct >= 50:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"  {skill:20s} {color}{correct:4.1f}/{total:<4d} ({pct:5.1f}%){Colors.ENDC}")
        total_correct += correct
        total_questions += total

    print(f"{'─'*40}")
    overall_pct = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"  {Colors.BOLD}{'OVERALL':20s} {total_correct:4.1f}/{total_questions:<4d} ({overall_pct:5.1f}%){Colors.ENDC}")

def print_failures(results, all_questions):
    """Print details of failed questions."""
    annotations = results.get("annotations", {})
    notes = results.get("notes", {})

    print(f"\n{Colors.BOLD}{Colors.RED}Failed Questions:{Colors.ENDC}")
    print(f"{'─'*50}")

    for q in all_questions:
        qid = q["id"]
        if qid in annotations and annotations[qid] == 0:
            print(f"\n{Colors.RED}✗ {qid}{Colors.ENDC}")
            print(f"  Q: {q['question'][:60]}...")
            print(f"  Expected: {q['correct']}")
            print(f"  Explanation: {q['explanation'][:80]}...")
            if qid in notes:
                print(f"  Note: {notes[qid]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate AI responses to NVFP4 GEMM questions")
    parser.add_argument("--results", "-r", help="Path to results JSON file")
    parser.add_argument("--summary", "-s", action="store_true", help="Just show summary of existing results")
    args = parser.parse_args()

    if args.summary and args.results:
        with open(args.results) as f:
            results = json.load(f)
        from questions import get_all_questions
        print_summary(results, get_all_questions())
        print_failures(results, get_all_questions())
    else:
        run_annotation_tui(args.results)

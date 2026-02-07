"""
GEMM Verifiers Environment

Wraps the NVFP4 GEMM evaluation (multiple-choice questions + code challenges)
into a verifiers-compatible environment for RL training.
"""

import sys
import os

# Add verifiers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verifiers'))

import verifiers as vf
from datasets import Dataset

from questions import QUESTIONS, get_all_questions
from questions_advanced import ADVANCED_QUESTIONS, get_advanced_questions
from code_challenges import CODE_CHALLENGES, get_code_challenges


def build_gemm_dataset(
    include_advanced: bool = True,
    include_code: bool = True,
) -> Dataset:
    """Build a HuggingFace Dataset from GEMM eval questions."""
    rows = []
    example_id = 0

    # Multiple-choice questions (core)
    for q in get_all_questions():
        prompt_text = _format_mc_prompt(q)
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "answer": q["correct"],
            "example_id": example_id,
            "task": "gemm-eval",
            "info": {
                "question_id": q["id"],
                "skill": q["skill"],
                "difficulty": q["difficulty"],
                "question_type": "multiple_choice",
                "explanation": q["explanation"],
                "reference": q.get("reference", ""),
            },
        })
        example_id += 1

    # Advanced questions
    if include_advanced:
        for q in get_advanced_questions():
            prompt_text = _format_mc_prompt(q)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "answer": q["correct"],
                "example_id": example_id,
                "task": "gemm-eval",
                "info": {
                    "question_id": q["id"],
                    "skill": q["skill"],
                    "difficulty": q["difficulty"],
                    "question_type": "multiple_choice",
                    "explanation": q["explanation"],
                    "reference": q.get("reference", ""),
                },
            })
            example_id += 1

    # Code challenges
    if include_code:
        for challenge in get_code_challenges():
            prompt_text = _format_code_prompt(challenge)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_CODE},
                    {"role": "user", "content": prompt_text},
                ],
                "answer": challenge["solution"],
                "example_id": example_id,
                "task": "gemm-eval-code",
                "info": {
                    "question_id": challenge["id"],
                    "skill": challenge["skill"],
                    "difficulty": challenge["difficulty"],
                    "question_type": "code_challenge",
                },
            })
            example_id += 1

    return Dataset.from_list(rows)


SYSTEM_PROMPT = (
    "You are an expert on NVIDIA Blackwell GPU architecture and NVFP4 GEMM kernel "
    "implementation. You have deep knowledge of TMA, tcgen05, TMEM, warp specialization, "
    "pipeline patterns, scale factor handling, and low-precision matrix multiplication.\n\n"
    "When answering multiple-choice questions, respond with ONLY the letter of your "
    "answer (A, B, C, or D) inside <answer> tags. Example: <answer>A</answer>"
)

SYSTEM_PROMPT_CODE = (
    "You are an expert on NVIDIA Blackwell GPU architecture and NVFP4 GEMM kernel "
    "implementation. When given a coding challenge, provide your solution inside "
    "<code> tags.\n\nExample:\n<code>\ndef my_function():\n    return 42\n</code>"
)


def _format_mc_prompt(question: dict) -> str:
    """Format a multiple-choice question as a prompt."""
    choices_str = "\n".join(question["choices"])
    return (
        f"**Question ({question['difficulty'].upper()}):**\n"
        f"{question['question']}\n\n"
        f"**Choices:**\n{choices_str}\n\n"
        f"Reply with ONLY the letter of your answer (A, B, C, or D) "
        f"inside <answer> tags."
    )


def _format_code_prompt(challenge: dict) -> str:
    """Format a code challenge as a prompt."""
    return (
        f"**Challenge ({challenge['difficulty'].upper()}) - {challenge['description']}:**\n\n"
        f"{challenge['prompt']}\n\n"
        f"Provide your complete solution inside <code> tags."
    )


def mc_reward(completion, answer, **kwargs) -> float:
    """
    Score a multiple-choice response.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    info = kwargs.get("info", {})
    if info.get("question_type") != "multiple_choice":
        return 0.0

    # Extract answer from <answer> tags
    response_text = ""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                response_text += msg.get("content", "")
    elif isinstance(completion, str):
        response_text = completion

    # Try to extract from <answer> tags
    import re
    match = re.search(r'<answer>\s*([A-Da-d])\s*</answer>', response_text)
    if match:
        given = match.group(1).upper()
        return 1.0 if given == answer.upper() else 0.0

    # Fallback: look for standalone letter at start
    clean = response_text.strip().upper()
    if not clean:
        return 0.0

    first_char = clean[0]
    if first_char in "ABCD":
        return 1.0 if first_char == answer.upper() else 0.0

    return 0.0


def format_reward(completion, **kwargs) -> float:
    """
    Reward for using the correct <answer> tag format.
    """
    info = kwargs.get("info", {})
    if info.get("question_type") != "multiple_choice":
        return 0.0

    response_text = ""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                response_text += msg.get("content", "")
    elif isinstance(completion, str):
        response_text = completion

    import re
    match = re.search(r'<answer>\s*[A-Da-d]\s*</answer>', response_text)
    return 1.0 if match else 0.0


def code_reward(completion, answer, **kwargs) -> float:
    """
    Score a code challenge response.
    Extracts code from <code> tags and validates using the challenge's test cases.
    """
    info = kwargs.get("info", {})
    if info.get("question_type") != "code_challenge":
        return 0.0

    response_text = ""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                response_text += msg.get("content", "")
    elif isinstance(completion, str):
        response_text = completion

    import re
    match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if not match:
        return 0.0

    submitted_code = match.group(1).strip()
    challenge_id = info.get("question_id", "")

    try:
        from code_challenges import validate_solution
        passed, _ = validate_solution(challenge_id, submitted_code)
        if passed is None:
            return 0.0
        return 1.0 if passed else 0.0
    except Exception:
        return 0.0


def load_environment(
    include_advanced: bool = True,
    include_code: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    Load the GEMM eval as a verifiers environment.

    Args:
        include_advanced: Include advanced questions (22 extra)
        include_code: Include code challenges (5 extra, requires torch for validation)
    """
    dataset = build_gemm_dataset(
        include_advanced=include_advanced,
        include_code=include_code,
    )

    reward_funcs = [mc_reward]
    reward_weights = [1.0]

    if include_code:
        reward_funcs.append(code_reward)
        reward_weights.append(1.0)

    # Format reward tracked as metric (weight=0)
    rubric = vf.Rubric(
        funcs=reward_funcs,
        weights=reward_weights,
    )
    rubric.add_reward_func(format_reward, weight=0.0)

    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )

    return env


# Allow direct testing
if __name__ == "__main__":
    print("Building GEMM eval dataset...")
    ds = build_gemm_dataset(include_advanced=True, include_code=True)
    print(f"Total examples: {len(ds)}")

    mc_count = sum(1 for r in ds if r["info"]["question_type"] == "multiple_choice")
    code_count = sum(1 for r in ds if r["info"]["question_type"] == "code_challenge")
    print(f"  Multiple-choice: {mc_count}")
    print(f"  Code challenges: {code_count}")

    # Show skill distribution
    from collections import Counter
    skills = Counter(r["info"]["skill"] for r in ds)
    print("\nBy skill:")
    for skill, count in sorted(skills.items()):
        print(f"  {skill}: {count}")

    difficulties = Counter(r["info"]["difficulty"] for r in ds)
    print("\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")

    # Test reward function
    print("\n--- Testing reward functions ---")
    test_completion = [{"role": "assistant", "content": "<answer>A</answer>"}]
    score = mc_reward(test_completion, "A", info={"question_type": "multiple_choice"})
    print(f"Correct answer test: {score}")  # Should be 1.0

    score = mc_reward(test_completion, "B", info={"question_type": "multiple_choice"})
    print(f"Wrong answer test: {score}")  # Should be 0.0

    fmt = format_reward(test_completion, info={"question_type": "multiple_choice"})
    print(f"Format reward test: {fmt}")  # Should be 1.0

    print("\nLoading as verifiers environment...")
    env = load_environment(include_code=False)
    print(f"Environment created: {type(env).__name__}")
    print(f"Dataset size: {len(env.get_dataset())}")

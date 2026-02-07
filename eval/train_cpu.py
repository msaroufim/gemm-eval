"""
Mini CPU-only RL Training Script for GEMM Eval

Implements a lightweight GRPO (Group Relative Policy Optimization) training loop
that works entirely on CPU using the GEMM verifiers environment.

This is a demonstration of the RL-for-evals pattern:
1. Sample completions from a small language model
2. Score them using the GEMM eval verifier
3. Compute advantages (reward - mean_reward per prompt)
4. Update the policy with REINFORCE-style gradients

Usage:
    source .venv/bin/activate
    cd eval
    python train_cpu.py --model gpt2 --max_steps 10 --rollouts_per_prompt 8
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import GEMM env components
sys.path.insert(0, os.path.dirname(__file__))
from questions import get_all_questions
from questions_advanced import get_advanced_questions
from gemm_env import mc_reward, format_reward, _format_mc_prompt, SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Mini CPU RL training on GEMM eval")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Number of training steps")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8,
                        help="Completions per prompt for GRPO")
    parser.add_argument("--prompts_per_step", type=int, default=2,
                        help="Number of unique prompts per training step")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max tokens to generate per completion")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Sampling temperature (higher = more exploration)")
    parser.add_argument("--include_advanced", action="store_true",
                        help="Include advanced questions")
    parser.add_argument("--output_dir", type=str, default="rl_outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompts(tokenizer, include_advanced=False):
    """Build tokenized prompts from GEMM eval questions."""
    questions = get_all_questions()
    if include_advanced:
        questions += get_advanced_questions()

    prompts = []
    for q in questions:
        user_text = _format_mc_prompt(q)

        # Format as chat if tokenizer supports it, otherwise raw text
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"{SYSTEM_PROMPT}\n\nUser: {user_text}\n\nAssistant:"
        else:
            text = f"{SYSTEM_PROMPT}\n\nUser: {user_text}\n\nAssistant:"

        prompts.append({
            "text": text,
            "answer": q["correct"],
            "question_id": q["id"],
            "skill": q["skill"],
            "difficulty": q["difficulty"],
        })

    return prompts


def generate_completions(model, tokenizer, prompt_text, n_completions, max_new_tokens, temperature):
    """Generate n completions for a prompt, returning tokens and log probs."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = input_ids.shape[1]

    completions = []
    for _ in range(n_completions):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 0.01),
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probs for the generated tokens
        all_scores = torch.stack(outputs.scores, dim=0)  # (gen_len, 1, vocab_size)
        log_probs_all = F.log_softmax(all_scores[:, 0, :], dim=-1)  # (gen_len, vocab_size)
        token_log_probs = log_probs_all[
            torch.arange(len(generated_ids)),
            generated_ids
        ]  # (gen_len,)

        completions.append({
            "text": generated_text,
            "input_ids": outputs.sequences[0].clone(),
            "prompt_len": prompt_len,
            "generated_ids": generated_ids.clone(),
            "token_log_probs": token_log_probs.clone(),
        })

    return completions


def score_completion(completion_text, correct_answer):
    """
    Score a completion with a multi-level reward:
    - 1.0: Correct answer in <answer> tags
    - 0.75: Correct letter found anywhere (without tags)
    - 0.5: Wrong answer but in <answer> tags (good format, wrong content)
    - 0.25: Any A/B/C/D letter present (trying to answer)
    - 0.0: No parseable answer at all
    """
    text = completion_text.strip()
    correct = correct_answer.upper()

    # Check for <answer>X</answer> format
    tag_match = re.search(r'<answer>\s*([A-Da-d])\s*</answer>', text)
    if tag_match:
        given = tag_match.group(1).upper()
        if given == correct:
            return 1.0, 1.0  # correct + good format
        else:
            return 0.5, 1.0  # wrong but good format

    # Check for standalone letter at very start
    first_char = text.upper()[:1] if text else ""
    if first_char in "ABCD":
        if first_char == correct:
            return 0.75, 0.0  # correct but no format
        else:
            return 0.25, 0.0  # wrong, no format, but tried

    # Check if any answer letter appears in the text
    for letter in "ABCD":
        # Look for patterns like "A)", "A.", "Answer: A", etc.
        if re.search(rf'\b{letter}\b', text.upper()):
            if letter == correct:
                return 0.5, 0.0
            else:
                return 0.1, 0.0

    return 0.0, 0.0


def compute_policy_loss(model, tokenizer, rollout_groups):
    """
    Compute GRPO-style policy gradient loss.

    For each group (same prompt, multiple completions):
    - advantage_i = reward_i - mean(rewards)
    - loss = -sum(advantage_i * log_prob_i)
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    total_tokens = 0

    for group in rollout_groups:
        rewards = torch.tensor([r["reward"] for r in group])
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward

        # Skip if all rewards are the same (no signal)
        if advantages.abs().max() < 1e-6:
            continue

        for rollout, advantage in zip(group, advantages):
            if abs(advantage.item()) < 1e-6:
                continue

            # Re-compute log probs with current policy (for on-policy gradient)
            input_ids = rollout["input_ids"].unsqueeze(0)
            prompt_len = rollout["prompt_len"]

            outputs = model(input_ids)
            logits = outputs.logits[0, prompt_len - 1:-1, :]  # Shift by 1 for next-token prediction
            log_probs = F.log_softmax(logits, dim=-1)

            gen_ids = rollout["generated_ids"]
            # Handle length mismatch
            min_len = min(log_probs.shape[0], len(gen_ids))
            if min_len == 0:
                continue

            token_log_probs = log_probs[:min_len, :].gather(
                1, gen_ids[:min_len].unsqueeze(1)
            ).squeeze(1)

            # REINFORCE loss: -advantage * sum(log_probs)
            rollout_loss = -(advantage * token_log_probs.sum())
            total_loss = total_loss + rollout_loss
            total_tokens += min_len

    if total_tokens > 0:
        total_loss = total_loss / total_tokens

    return total_loss


def plot_curves(log, output_dir):
    """Plot training curves and save to output_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots. Install with: uv pip install matplotlib")
        return

    steps_data = log["steps"]
    steps = [s["step"] for s in steps_data]
    rewards = [s["mean_reward"] for s in steps_data]
    format_rewards = [s["mean_format_reward"] for s in steps_data]
    losses = [s["loss"] for s in steps_data]
    grad_norms = [s.get("grad_norm", 0.0) for s in steps_data]
    any_letter_rates = [s.get("any_letter_rate", 0.0) for s in steps_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GEMM Eval RL Training (CPU, GRPO)", fontsize=14)

    # Reward
    ax = axes[0, 0]
    ax.plot(steps, rewards, 'b-o', linewidth=2, markersize=4, label='Mean Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Mean Reward per Step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Loss
    ax = axes[0, 1]
    ax.plot(steps, losses, 'r-o', linewidth=2, markersize=4, label='Policy Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Gradient Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format reward / any-letter rate
    ax = axes[1, 0]
    ax.plot(steps, format_rewards, 'g-o', linewidth=2, markersize=4, label='Format (<answer> tags)')
    ax.plot(steps, any_letter_rates, 'm-s', linewidth=2, markersize=4, label='Any letter (A/B/C/D)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Rate')
    ax.set_title('Format Compliance')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Grad norm
    ax = axes[1, 1]
    ax.plot(steps, grad_norms, 'k-o', linewidth=2, markersize=4, label='Gradient Norm')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = Path(output_dir) / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")


def print_ascii_curves(log):
    """Print ASCII training curves to terminal."""
    steps_data = log["steps"]
    rewards = [s["mean_reward"] for s in steps_data]
    losses = [s["loss"] for s in steps_data]

    width = 50

    print("\n=== Reward Curve ===")
    max_r = max(rewards) if max(rewards) > 0 else 1.0
    for i, r in enumerate(rewards):
        bar_len = int(r / max_r * width) if max_r > 0 else 0
        print(f"  Step {i+1:3d} | {'█' * bar_len}{'░' * (width - bar_len)} | {r:.4f}")

    print("\n=== Loss Curve ===")
    max_l = max(abs(l) for l in losses) if any(l != 0 for l in losses) else 1.0
    for i, l in enumerate(losses):
        bar_len = int(abs(l) / max_l * width) if max_l > 0 else 0
        print(f"  Step {i+1:3d} | {'█' * bar_len}{'░' * (width - bar_len)} | {l:.4f}")


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Mini CPU RL Training for GEMM Eval ===")
    print(f"Model: {args.model}")
    print(f"Steps: {args.max_steps}")
    print(f"Rollouts/prompt: {args.rollouts_per_prompt}")
    print(f"Prompts/step: {args.prompts_per_step}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print()

    # Load model and tokenizer
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # CPU needs float32
    )
    model.train()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build prompts
    prompts = build_prompts(tokenizer, include_advanced=args.include_advanced)
    print(f"Total eval prompts: {len(prompts)}")
    print()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training log
    log = {
        "config": vars(args),
        "steps": [],
    }

    prompt_idx = 0
    for step in range(args.max_steps):
        step_start = time.time()
        print(f"--- Step {step + 1}/{args.max_steps} ---")

        # Select prompts for this step (cycle through)
        step_prompts = []
        for _ in range(args.prompts_per_step):
            step_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        # Generate rollouts
        rollout_groups = []
        step_rewards = []
        step_format_rewards = []
        step_has_letter = 0
        step_total = 0

        for prompt_data in step_prompts:
            print(f"  Generating {args.rollouts_per_prompt} completions for {prompt_data['question_id']}...")

            completions = generate_completions(
                model, tokenizer, prompt_data["text"],
                n_completions=args.rollouts_per_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            group = []
            for comp in completions:
                reward, fmt_reward = score_completion(comp["text"], prompt_data["answer"])
                comp["reward"] = reward
                comp["format_reward"] = fmt_reward
                group.append(comp)

                step_rewards.append(reward)
                step_format_rewards.append(fmt_reward)
                step_total += 1
                # Track if any letter A-D appears
                if re.search(r'[A-Da-d]', comp["text"]):
                    step_has_letter += 1

            rollout_groups.append(group)

            # Show sample completions
            sorted_group = sorted(group, key=lambda x: x["reward"], reverse=True)
            best = sorted_group[0]
            worst = sorted_group[-1]
            print(f"    Best  (r={best['reward']:.2f}): {best['text'][:80].replace(chr(10), ' ')}...")
            print(f"    Worst (r={worst['reward']:.2f}): {worst['text'][:80].replace(chr(10), ' ')}...")

        # Compute advantages and update policy
        print("  Computing policy gradient...")
        optimizer.zero_grad()
        loss = compute_policy_loss(model, tokenizer, rollout_groups)

        grad_norm_val = 0.0
        has_gradient = loss.requires_grad and abs(loss.item()) > 1e-8
        if has_gradient:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_val = grad_norm.item()
            optimizer.step()
            print(f"  Loss: {loss.item():.6f}, Grad norm: {grad_norm_val:.4f}")
        else:
            print("  No gradient signal (all rewards equal within groups)")

        # Step metrics
        mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
        mean_format = sum(step_format_rewards) / len(step_format_rewards) if step_format_rewards else 0
        any_letter_rate = step_has_letter / step_total if step_total > 0 else 0
        step_time = time.time() - step_start

        # Reward distribution
        reward_counts = defaultdict(int)
        for r in step_rewards:
            reward_counts[r] += 1

        step_log = {
            "step": step + 1,
            "mean_reward": mean_reward,
            "mean_format_reward": mean_format,
            "any_letter_rate": any_letter_rate,
            "loss": loss.item() if loss.requires_grad else 0.0,
            "grad_norm": grad_norm_val,
            "num_rollouts": len(step_rewards),
            "reward_distribution": {str(k): v for k, v in sorted(reward_counts.items())},
            "time_seconds": round(step_time, 1),
            "prompts": [p["question_id"] for p in step_prompts],
        }
        log["steps"].append(step_log)

        print(f"  Mean reward: {mean_reward:.4f} | Format: {mean_format:.3f} | "
              f"Any letter: {any_letter_rate:.1%}")
        print(f"  Reward dist: {dict(sorted(reward_counts.items()))}")
        print(f"  Step time: {step_time:.1f}s")
        print()

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Save model checkpoint
    ckpt_path = output_dir / "final_model"
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # Print ASCII curves (always works)
    print_ascii_curves(log)

    # Plot with matplotlib if available
    plot_curves(log, output_dir)

    # Print summary
    print("\n=== Training Summary ===")
    print(f"Steps completed: {args.max_steps}")
    total_rollouts = sum(s["num_rollouts"] for s in log["steps"])
    print(f"Total rollouts: {total_rollouts}")
    rewards = [s["mean_reward"] for s in log["steps"]]
    print(f"Reward trajectory: {' -> '.join(f'{r:.4f}' for r in rewards)}")
    losses = [s["loss"] for s in log["steps"]]
    print(f"Loss trajectory:   {' -> '.join(f'{l:.4f}' for l in losses)}")

    return log


if __name__ == "__main__":
    args = parse_args()
    train(args)

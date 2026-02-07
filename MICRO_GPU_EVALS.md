# Integrating Micro GPU Evals into RL Training

This document explains how to take a domain-specific evaluation (like the NVFP4 GEMM eval) and integrate it into an RL training pipeline using [verifiers](https://github.com/PrimeIntellect-ai/verifiers) and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Architecture Overview

The pipeline has three layers:

```
┌─────────────────────────────────────────────────────────┐
│                     prime-rl                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Inference │  │ Orchestrator │  │     Trainer      │  │
│  │  (vLLM)  │←→│  (env loop)  │──│ (GRPO/AIPO loss) │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
│                       ↑                                 │
│                       │ uses                            │
│               ┌───────┴────────┐                        │
│               │   verifiers    │                        │
│               │  Environment   │                        │
│               └───────┬────────┘                        │
│                       │ wraps                           │
│               ┌───────┴────────┐                        │
│               │  Your Eval     │                        │
│               │ (questions,    │                        │
│               │  reward fn)    │                        │
│               └────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

**Inference** generates completions from the model using vLLM.
**Orchestrator** feeds prompts from the environment, collects completions, scores them, and ships training batches.
**Trainer** updates model weights using the advantages computed from rewards.

## Step 1: Design Your Eval

A micro GPU eval is a focused, domain-specific evaluation that tests whether a model understands a narrow technical domain. Good candidates:

- **Multiple-choice questions** covering key concepts (easy to score, clear signal)
- **Code challenges** with automated validation (richer signal, harder to score)
- **Structured output tasks** where you can verify correctness programmatically

The GEMM eval tests knowledge of NVFP4 GEMM kernel implementation across 16 skills with 56 multiple-choice questions and 5 code challenges.

### Requirements for RL compatibility

1. **Deterministic scoring**: Your reward function must produce the same score for the same output. No LLM-as-judge unless you can tolerate noise.
2. **Binary or near-binary rewards**: Clean signal works best for RL. Correct=1.0, wrong=0.0. Partial credit is fine but keep it simple.
3. **Enough questions**: You need sufficient diversity to avoid overfitting. 50+ questions is a good starting point.
4. **Varied difficulty**: Mix of easy/medium/hard prevents reward hacking on trivial questions.

## Step 2: Create a Verifiers Environment

The verifiers framework provides the `Environment` base class. For most evals, `SingleTurnEnv` is the right choice (one prompt, one response).

### Minimal example

```python
import verifiers as vf
from datasets import Dataset
import re

def load_environment(**kwargs) -> vf.Environment:
    # 1. Build dataset with required columns
    rows = []
    for q in YOUR_QUESTIONS:
        rows.append({
            "prompt": [
                {"role": "system", "content": "You are an expert on X..."},
                {"role": "user", "content": format_question(q)},
            ],
            "answer": q["correct"],      # ground truth
            "example_id": q["id"],        # unique int
            "task": "your-eval-name",     # string identifier
            "info": {"skill": q["skill"], "question_type": "mc"},
        })
    dataset = Dataset.from_list(rows)

    # 2. Define reward function
    def correctness_reward(completion, answer, **kwargs) -> float:
        # Extract model's answer from completion
        text = ""
        for msg in completion:
            if msg.get("role") == "assistant":
                text += msg.get("content", "")
        match = re.search(r'<answer>\s*([A-D])\s*</answer>', text)
        if match:
            return 1.0 if match.group(1) == answer else 0.0
        return 0.0

    # 3. Create rubric with reward functions
    rubric = vf.Rubric(funcs=[correctness_reward], weights=[1.0])

    # 4. Return environment
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
```

### Key design decisions

**System prompt**: Tell the model to format its answer in a parseable way (XML tags like `<answer>A</answer>` work well). This makes scoring reliable.

**Reward functions** receive these kwargs automatically: `completion`, `answer`, `prompt`, `task`, `info`, `state`. Use `info` to pass metadata from your dataset.

**Format rewards**: Track format compliance as a metric (weight=0.0) to monitor whether the model learns to use the expected format:

```python
rubric.add_reward_func(format_reward, weight=0.0)  # tracked but doesn't affect training
```

**Multiple reward signals**: Use `weights` to combine rewards. Weight=0.0 means "track as metric only".

### File structure for a verifiers environment package

```
your_eval/
├── your_eval.py         # Must contain load_environment(**kwargs) -> Environment
├── pyproject.toml       # Package metadata
└── README.md
```

`pyproject.toml`:
```toml
[project]
name = "your-eval"
version = "0.1.0"
tags = ["gpu", "kernels", "single-turn"]
description = "Your eval description"
dependencies = ["verifiers>=0.1.5"]
```

## Step 3: Test Your Environment Locally

Before hooking into RL training, verify the environment works standalone:

```python
import asyncio
from openai import AsyncOpenAI

env = load_environment()
dataset = env.get_dataset()
print(f"Dataset size: {len(dataset)}")

# Test with a real model
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
results = asyncio.run(env.generate(
    inputs=dataset.select(range(5)),
    client=client,
    model="your-model",
    max_concurrent=4,
))

for r in results:
    print(f"Reward: {r['reward']}, Completion: {r['completion'][-1]['content'][:50]}")
```

## Step 4: Run RL Training with prime-rl (GPU)

Once your environment works, integrate with prime-rl for full distributed RL training.

### Configuration

Create a TOML config:

```toml
# rl_config.toml
output_dir = "outputs/gemm_rl"
max_steps = 100
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

[model]
name = "Qwen/Qwen3-4B"

[orchestrator]
batch_size = 64
seq_len = 2048
rollouts_per_example = 8

[orchestrator.sampling]
temperature = 0.7
max_tokens = 64

[[orchestrator.env]]
id = "gemm-eval"          # your environment id
args = {include_advanced = true}

[trainer.data]
# uses real data from orchestrator

[inference]
# uses defaults
```

### Running

```bash
uv run rl @ rl_config.toml
```

This starts three processes:
1. **vLLM inference server** on GPU 0
2. **Orchestrator** on CPU, using your verifiers environment to score completions
3. **Trainer** on GPU 1, updating model weights with GRPO/AIPO loss

### Using verifiers' built-in RL trainer (alternative)

If you don't need prime-rl's full distributed infrastructure:

```python
from verifiers.rl import RLTrainer, RLConfig

config = RLConfig(
    model="Qwen/Qwen3-4B",
    run_name="gemm-rl",
    use_lora=True,
    lora_rank=8,
    learning_rate=1e-5,
    micro_batch_size=4,
    rollouts_per_example=8,
    batch_size=256,
    max_steps=100,
    max_tokens=64,
    max_seq_len=1024,
    temperature=0.7,
)

env = load_environment(include_advanced=True)
trainer = RLTrainer(model=config.model, env=env, args=config)
trainer.train()
```

## Step 5: CPU-Only Development Loop

For development and testing without GPUs, use the standalone training script pattern (see `eval/train_cpu.py`):

```python
# Core loop: generate -> score -> compute advantage -> update
for step in range(max_steps):
    for prompt in batch:
        # Generate N completions per prompt
        completions = model.generate(prompt, n=rollouts_per_prompt)

        # Score each completion with your reward function
        rewards = [reward_fn(c, answer) for c in completions]

        # GRPO advantage: reward_i - mean(rewards)
        mean_r = mean(rewards)
        advantages = [r - mean_r for r in rewards]

        # Policy gradient: -advantage * log_prob(completion)
        for comp, adv in zip(completions, advantages):
            loss += -adv * sum(log_probs(comp))

    loss.backward()
    optimizer.step()
```

This won't produce a useful model (GPT-2 doesn't know about GEMM kernels), but it validates that your reward function produces meaningful signal and the training loop is correct.

## Key Principles for Micro GPU Eval Design

### 1. Make rewards verifiable, not judgmental

Bad: "Is this a good explanation of TMA?" (requires LLM judge)
Good: "Which letter matches the correct answer?" (deterministic comparison)

### 2. Use structured output formats

Require the model to put its answer in parseable tags:
```
<answer>B</answer>
```
This eliminates parsing ambiguity and gives you a clean format reward signal.

### 3. Cover the skill tree, not just the surface

The GEMM eval covers 16 distinct skills across 3 difficulty levels. This prevents the model from reward-hacking by memorizing a few patterns. Structure your eval around a skill taxonomy:

```
Domain
├── Skill 1 (easy, medium, hard)
├── Skill 2 (easy, medium, hard)
├── ...
└── Skill N (easy, medium, hard)
```

### 4. Include both recall and application

- **Multiple-choice**: Tests recall and conceptual understanding
- **Code challenges**: Tests ability to apply knowledge (harder to score, richer signal)

### 5. Reference real artifacts

Each question should reference specific code files and line numbers. This grounds the eval in real implementations and makes it auditable.

### 6. Start small, iterate fast

1. Start with 20-30 multiple-choice questions covering core skills
2. Run a few RL steps to see if the reward signal is meaningful
3. Add more questions and difficulty levels based on what the model gets wrong
4. Add code challenges once the MC questions show learning

### 7. Scale considerations

| Setup | Inference | Training | Use case |
|-------|-----------|----------|----------|
| CPU only | HF generate | Basic REINFORCE | Development, testing |
| 1 GPU | vLLM | LoRA + GRPO | Prototyping |
| 2+ GPUs | vLLM (TP) | FSDP2 + AIPO | Production training |
| Multi-node | prime-rl distributed | prime-rl distributed | Large-scale runs |

## Files in This Repo

| File | Description |
|------|-------------|
| `eval/gemm_env.py` | Verifiers environment wrapping the GEMM eval |
| `eval/train_cpu.py` | Standalone CPU-only RL training script |
| `eval/questions.py` | 34 core multiple-choice questions (8 skills) |
| `eval/questions_advanced.py` | 22 advanced questions (8 more skills) |
| `eval/code_challenges.py` | 5 code challenges with validation |
| `eval/runner.py` | Original eval runner (batch/interactive modes) |

## Quick Start

```bash
# 1. Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e verifiers/
uv pip install torch transformers datasets

# 2. Test the environment
cd eval && python gemm_env.py

# 3. Run CPU training demo
python train_cpu.py --model gpt2 --max_steps 3

# 4. For GPU training with prime-rl
# See Step 4 above for configuration
```

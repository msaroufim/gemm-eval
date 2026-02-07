# NVFP4 GEMM: Kernels, Eval, and RL Training

Expert NVFP4 GEMM kernel implementations for NVIDIA Blackwell (SM100) GPUs, a skill evaluation framework, and an RL training pipeline that uses the eval as a reward signal.

## What's here

```
.
├── 1.py - 6.py              # Expert NVFP4 GEMM kernels (fastest to slowest)
├── skills.md                 # Complete guide to kernel implementation patterns
├── eval/                     # Evaluation framework + RL training
│   ├── questions.py          # 34 core MC questions across 8 skills
│   ├── questions_advanced.py # 22 advanced MC questions across 8 more skills
│   ├── code_challenges.py    # 5 code challenges with automated validation
│   ├── runner.py             # Eval runner, scoring, reports
│   ├── run_eval.py           # CLI entry point (export/import/interactive/annotate)
│   ├── gemm_env.py           # Verifiers environment (wraps eval for RL)
│   ├── train_cpu.py          # Standalone CPU-only GRPO training script
│   └── tui.py                # Terminal UI for manual annotation
├── MICRO_GPU_EVALS.md        # Guide: integrating micro GPU evals into RL
├── verifiers/                # git clone of PrimeIntellect/verifiers
└── prime-rl/                 # git clone of PrimeIntellect/prime-rl
```

## What we built

1. **GEMM verifiers environment** (`eval/gemm_env.py`) — wraps the 56 MC questions + 5 code challenges as a `vf.SingleTurnEnv` with a multi-level reward function
2. **CPU-only GRPO training script** (`eval/train_cpu.py`) — standalone RL loop, no GPU needed
3. **Training curves** (`eval/rl_outputs/training_curves.png`) — 10 steps, 160 rollouts on GPT-2
4. **Integration guide** (`MICRO_GPU_EVALS.md`) — how to turn any micro eval into an RL reward signal

## Quick Start

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e verifiers/
uv pip install torch transformers datasets matplotlib

cd eval
```

### Run the eval standalone

```bash
python run_eval.py list                                    # list skills
python run_eval.py export -o prompts.json                  # export for batch eval
python run_eval.py import -p prompts.json -r responses.json  # score responses
```

### Run CPU RL training (GPT-2 demo)

```bash
python train_cpu.py --model gpt2 --max_steps 10 --rollouts_per_prompt 8 --temperature 1.5
```

Produces ASCII curves in terminal + `rl_outputs/training_curves.png` + `rl_outputs/training_log.json`.

GPT-2 has no GEMM knowledge so reward stays low (~0.1-0.3 from partial credit on random letter hits), but the full GRPO pipeline runs: generate → score → compute advantages → policy gradient → update.

## Next: Try a Fancier Model

The CPU script works with any HuggingFace model. Bigger models will be slow on CPU but will actually have some coding/technical knowledge to surface.

### Option 1: Qwen2.5-0.5B on CPU (slow but real signal)

```bash
python train_cpu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max_steps 5 \
  --rollouts_per_prompt 8 \
  --prompts_per_step 2 \
  --max_new_tokens 32 \
  --temperature 0.8 \
  --lr 1e-5
```

This model has a chat template and actual instruction-following ability, so it should produce `<answer>A</answer>` format and get some questions right.

### Option 2: Qwen2.5-1.5B on CPU (very slow, better signal)

```bash
python train_cpu.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max_steps 3 \
  --rollouts_per_prompt 4 \
  --prompts_per_step 1 \
  --max_new_tokens 32 \
  --temperature 0.7 \
  --lr 5e-6
```

### Option 3: GPU training with verifiers' built-in RL trainer (1 GPU)

```bash
uv pip install vllm peft liger-kernel wandb

# Terminal 1: start vLLM inference server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --enforce-eager

# Terminal 2: run RL training
python -c "
import sys; sys.path.insert(0, 'eval')
from gemm_env import load_environment
from verifiers.rl import RLTrainer, RLConfig

env = load_environment(include_advanced=True)
config = RLConfig(
    model='Qwen/Qwen2.5-3B-Instruct',
    run_name='gemm-rl',
    use_lora=True,
    lora_rank=8,
    learning_rate=1e-5,
    micro_batch_size=4,
    rollouts_per_example=8,
    batch_size=128,
    max_steps=50,
    max_tokens=64,
    max_seq_len=1024,
    temperature=0.7,
    vllm_server_port=8000,
)
trainer = RLTrainer(model=config.model, env=env, args=config)
trainer.train()
"
```

### Option 4: Full distributed training with prime-rl (2+ GPUs)

Create `rl_config.toml`:

```toml
output_dir = "outputs/gemm_rl"
max_steps = 100
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

[model]
name = "Qwen/Qwen2.5-3B-Instruct"

[orchestrator]
batch_size = 64
rollouts_per_example = 8
seq_len = 1024

[orchestrator.sampling]
temperature = 0.7
max_tokens = 64

[[orchestrator.env]]
id = "gemm-eval"
args = {include_advanced = true}
```

```bash
cd prime-rl
uv run rl @ ../rl_config.toml
```

## How the reward works

The `score_completion` function in `train_cpu.py` uses multi-level rewards to create gradient signal even from weak models:

| Score | Condition |
|-------|-----------|
| 1.0 | Correct answer in `<answer>X</answer>` tags |
| 0.75 | Correct letter at start of response (no tags) |
| 0.5 | Wrong answer in `<answer>` tags, or correct letter found in text |
| 0.25 | Wrong letter at start (tried to answer) |
| 0.1 | Wrong letter found somewhere in text |
| 0.0 | No parseable answer |

GRPO computes advantage = reward_i - mean(group_rewards). When some completions score 0.25 and others score 0.0, the 0.25 ones get positive advantage and the model learns to at least produce a letter. This bootstraps toward actually answering correctly.

## The eval

56 multiple-choice questions + 5 code challenges across 16 skills and 3 difficulty levels:

| Area | Skills | Questions |
|------|--------|-----------|
| Core | TMA, Async Copy, tcgen05, TMEM, Scale Factors, Warp Specialization, Pipelines, Integration | 34 |
| Advanced | TMA Advanced, UMMA, TMEM Advanced, WGMMA, Quantization, Heuristics, MoE Patterns, Clusters | 22 |
| Code | Scale factor permutation, barrier patterns, warp assignment, pipeline index, TMEM layout | 5 |

## How the RL pipeline works

```
prompt from eval  →  model generates N completions  →  score each one (0.0 - 1.0)
                                                              ↓
model weights updated  ←  GRPO loss  ←  advantage = reward_i - mean(rewards)
```

GRPO doesn't need a value function. It generates multiple completions per prompt, scores them, and uses group-relative advantage as the training signal. Completions better than the group average get reinforced; worse ones get suppressed.

## Integration Guide

See `MICRO_GPU_EVALS.md` for how to turn any domain-specific eval into a verifiers environment and hook it into RL training.

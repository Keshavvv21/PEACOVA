"""Nightly local-SLM fine-tuning — stage 2 of the pipeline (stage 1:
agent/finetune_log.py's data collection, curated by data_engine.py — see
below). Deliberately a STANDALONE script with its own requirements file
(finetune_requirements.txt), never imported by the live chatbot app —
torch/transformers/peft/datasets are too heavy to add to the
always-running serving image. Run this in its own venv/container; see
deploy/nightly_finetune.sh for how the actual nightly cron invokes it.

What this does, in order:
  1. Read the JSONL log (agent/finetune_log.py's output) — each line is
     one (messages, response, reward) example from a real production
     turn the local model actually answered.
  2. Filter to --min-reward and cap at --max-examples (most recent
     first) — a small, high-signal training set, not everything ever
     logged.
  2b. Data Engine (data_engine.py) — scrub PII/credential-shaped
     substrings from every example BEFORE it can be baked into weights,
     then check each one against a DURABLE curated corpus
     (--curated-path, growing across every night this has ever run, not
     just tonight's raw log) and keep only what's genuinely new. The
     actual training set below is the curated store's full accumulated
     contents, not just tonight's batch — pass --no-data-engine to fall
     back to the old behavior (train on tonight's filtered raw examples
     directly) for a quick local test run.
  3. Render each kept example through the target model's OWN chat
     template (so training input matches inference input byte-for-byte)
     and mask the loss to the assistant's final turn only — the model
     is being taught to answer well, not to reproduce its own system
     prompt/tool schema back verbatim.
  4. LoRA fine-tune (peft), reward-weighted: each example's loss is
     scaled by its own reward before backprop, so a 1.0-reward example
     teaches more than a 0.7-reward one, not identically — the
     "RL-informed" half of "reward-weighted supervised fine-tuning," a
     well-established lightweight stand-in for full online RL (PPO-style
     rollouts), which is not remotely practical on a CPU-only box.
     --qlora loads the base model 4-bit (bitsandbytes) instead of the
     default full-precision LoRA — a real, opt-in path for a box with a
     CUDA GPU; refuses cleanly (not a cryptic crash) if requested
     without one, since bitsandbytes' 4-bit kernels need CUDA.
  5. Bounded by --max-minutes (wall-clock) as the real safety cutoff,
     not --max-steps alone — CPU training speed varies enough by
     machine that a step count doesn't reliably bound wall-clock time
     on an unattended nightly job.
  6. Merge the LoRA adapter into the base weights. Optionally,
     --prune-sparsity applies real global L1-unstructured magnitude
     pruning (torch's own torch.nn.utils.prune, no new dependency) to
     the merged model's linear layers — genuinely zeroes that fraction
     of weights, verified by an actual sparsity measurement printed
     after. Honest limitation: GGUF/Q4_K_M quantization (the next
     script) does not specially exploit sparse weights for size or
     speed today, so this is a real, measurable transform without a
     proven inference-time win through THIS serving path yet — left
     off (0.0) by default for that reason.
  7. Save the result — a plain HF model directory. GGUF conversion/
     quantization and registering the result with the live chatbot are
     the NEXT script (convert_and_register_ollama_model.sh) —
     llama.cpp and an admin API call are a different kind of dependency
     than this script's.

Validated: the mechanics here (data loading/filtering, the chat-
template rendering + loss masking, the reward-weighted loss, the
time-boxed stop, the LoRA merge/save) were smoke-tested end-to-end
against the real Qwen2.5-0.5B-Instruct checkpoint (same architecture/
tokenizer/chat-template family as the production 1.5B/3B models, just
the smallest member) — not against a CPU-only multi-hour run on the
actual 1.5B/3B model, which needs the real deploy box to validate for
real. Re-run once there with real accumulated data before trusting the
output unattended. The pruning and QLoRA paths are new and share that
same caveat, doubly so for QLoRA since it needs a CUDA box this
CPU-only pipeline was never validated against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# CPU thread cap — BEFORE `import torch`, because PyTorch's native BLAS/
# OpenMP thread pools (OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS)
# are sized once at their FIRST use, not reconfigurable afterward the same
# way torch.set_num_threads() is. Without this, torch defaults to ALL
# visible cores for its own intra-op parallelism — on an unattended nightly
# box that's also running the live ak-backend/ak-chatbot containers, that's
# a real "why is everything else slow right now" problem, not a hypothetical
# one (`docker run`'s own `--cpus` cap in deploy/nightly_finetune.sh is the
# other, OS-level half of this fix — this half matters for a plain venv run
# too, e.g. the docstring's own "--no-data-engine ... for a quick local test
# run" path, which never goes through Docker at all).
# Leaves ONE core free for the rest of the system by default, same
# "leave headroom, don't claim every last resource" reasoning
# provider_optimizer.py's own _SAVE_EVERY throttling uses for its DB writes.
_CPU_THREADS = int(os.getenv("FINETUNE_CPU_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_THREADS))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

torch.set_num_threads(_CPU_THREADS)  # belt-and-suspenders: some ops read
                                      # this directly rather than the env vars above

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_engine  # noqa: E402 — after the sys.path fix-up above


def load_examples(log_path: Path, *, min_reward: float, max_examples: int) -> list[dict]:
    """Reads the JSONL log, keeps only reward >= min_reward, most-recent
    --max-examples first. A missing/unreadable log => empty list (a
    quiet, correct "nothing to train on yet" rather than a crash — this
    runs unattended every night, and "no new data since last night" is
    the expected common case, not an error)."""
    if not log_path.exists():
        return []
    kept: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("reward", 0.0) >= min_reward:
                kept.append(record)
    kept.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return kept[:max_examples]


def build_dataset(examples: list[dict], tokenizer, *, max_length: int) -> Dataset:
    """Renders each example through the tokenizer's own chat template
    and masks the loss to the assistant's final turn only.

    The per-example loss scale is reward * training_weight, not reward
    alone — "training_weight" (default 1.0, so every example logged
    before this field existed is unaffected) is agent/finetune_log.py's
    log_ground_truth_turn() knob for "train harder on THIS example
    specifically," e.g. a curated complex/L3-L4 prompt worth more
    gradient signal per epoch than a routine one of equal reward. Reward
    still means "was this answer correct" and training_weight still
    means "how much should a correct example of this kind count" — kept
    as two separate fields end-to-end (this is the only place they're
    multiplied together) rather than conflating them at the logging
    layer, so a future consumer of the raw log can still tell the two
    apart."""
    input_ids_list, labels_list, rewards_list = [], [], []
    for ex in examples:
        convo = ex["messages"] + [{"role": "assistant", "content": ex["response"]}]
        # apply_chat_template(tokenize=True) returns a BatchEncoding, not
        # a plain list — ["input_ids"] is the actual token id list.
        full_ids = tokenizer.apply_chat_template(convo, tokenize=True,
                                                  add_generation_prompt=False)["input_ids"]
        prompt_ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True,
                                                    add_generation_prompt=True)["input_ids"]
        full_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        rewards_list.append(float(ex.get("reward", 1.0)) * float(ex.get("training_weight", 1.0)))
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list,
                              "reward": rewards_list})


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask, rewards = [], [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
        rewards.append(b["reward"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "reward": torch.tensor(rewards, dtype=torch.float),
    }


class RewardWeightedTrainer(Trainer):
    """The one real departure from plain supervised fine-tuning: scales
    each example's loss by its own reward before backprop, so a
    perfect-reward example teaches more than a partial-reward one
    instead of identically — reward-weighted regression, the practical
    stand-in for full RLHF/PPO on hardware that can't run real rollouts."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = inputs.pop("reward")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        labels = inputs["labels"][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        per_token_loss = per_token_loss.view(labels.size())
        mask = (labels != -100).float()
        per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        weighted = (per_example_loss * rewards).sum() / rewards.sum().clamp(min=1e-6)
        return (weighted, outputs) if return_outputs else weighted


class WallClockStop(TrainerCallback):
    """--max-minutes is the real safety cutoff, not --max-steps alone —
    CPU training speed varies enough by machine that a step count
    doesn't reliably bound wall-clock time on an unattended nightly job."""

    def __init__(self, max_minutes: float):
        self.deadline = time.monotonic() + max_minutes * 60

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() > self.deadline:
            control.should_training_stop = True
        return control


def prune_model(model: torch.nn.Module, sparsity: float) -> float:
    """Real, verifiable global L1-unstructured magnitude pruning over
    every torch.nn.Linear layer's weight — torch's own
    torch.nn.utils.prune, no new dependency. `sparsity` is the fraction
    of each layer's weights to zero (the smallest-magnitude ones, which
    matter least to the layer's output). prune.remove() makes it
    permanent (folds the pruning mask into the actual weight tensor, so
    it survives save_pretrained() as plain zeros, not a separate mask
    peft/transformers wouldn't know to reload). Returns the ACTUAL
    measured sparsity afterward — a real number to print/verify, not a
    claim.

    Honest limitation (see this file's own module docstring): the next
    stage of this pipeline quantizes to GGUF/Q4_K_M for serving, and
    that quantizer does not specially exploit exact zeros for a smaller
    or faster result today — so this is a genuine, measured transform
    without a proven win through the actual serving path yet. Left at
    0.0 (a no-op) by default; opt in with --prune-sparsity only if
    you're specifically experimenting with it.
    """
    if sparsity <= 0.0:
        return 0.0
    import torch.nn.utils.prune as prune

    linear_layers = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if not linear_layers:
        return 0.0
    prune.global_unstructured(linear_layers, pruning_method=prune.L1Unstructured, amount=sparsity)
    for module, name in linear_layers:
        prune.remove(module, name)

    zeros = sum(int((m.weight == 0).sum()) for m, _ in linear_layers)
    total = sum(m.weight.numel() for m, _ in linear_layers)
    return zeros / total if total else 0.0


def load_base_model(base_model: str, *, qlora: bool):
    """Plain full-precision load by default (torch_dtype=float32, the
    existing CPU-friendly path). --qlora loads the SAME base model
    4-bit via bitsandbytes instead — a real, opt-in path, but one that
    needs an actual CUDA GPU: bitsandbytes' 4-bit kernels don't run on
    CPU. Refuses with a clear message rather than a cryptic crash deep
    inside bitsandbytes if requested without one."""
    if not qlora:
        return AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

    if not torch.cuda.is_available():
        raise SystemExit(
            "--qlora requires a CUDA GPU (bitsandbytes' 4-bit kernels don't run on CPU) — "
            "this box has none available to torch. Drop --qlora to use the default "
            "full-precision LoRA path, which IS CPU-friendly.")
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise SystemExit(
            "--qlora needs bitsandbytes installed (pip install bitsandbytes) — it's "
            "deliberately not in finetune_requirements.txt by default since it's only "
            "useful on a CUDA box, unlike everything else in that file.") from e

    from peft import prepare_model_for_kbit_training

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map="auto")
    return prepare_model_for_kbit_training(model)


def main() -> None:
    print(f"CPU threads capped at {_CPU_THREADS} (of {os.cpu_count()} visible) — "
         f"override with FINETUNE_CPU_THREADS")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-path", default="data/finetune/ollama_turns.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="data/finetune/merged")
    parser.add_argument("--min-reward", type=float, default=0.7)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-minutes", type=float, default=120.0,
                        help="Hard wall-clock cutoff — the real safety bound on a nightly job.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--min-examples", type=int, default=8,
                        help="Below this, skip training tonight rather than overfit on scraps.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Qwen2's attention+MLP projection names by default — override for a "
                             "different base-model architecture.")
    parser.add_argument("--curated-path", default="data/finetune/curated_examples.jsonl",
                        help="The Data Engine's durable, ever-growing curated corpus "
                             "(data_engine.DataEngineStore) — see --no-data-engine to bypass it.")
    parser.add_argument("--no-data-engine", action="store_true",
                        help="Skip PII scrubbing + the persistent curated store; train directly "
                             "on tonight's --min-reward-filtered raw log, like before this existed. "
                             "For a quick local test run only — the nightly cron should not set this.")
    parser.add_argument("--dedup-threshold", type=float, default=0.95,
                        help="Cosine similarity above which two examples count as duplicates.")
    parser.add_argument("--prune-sparsity", type=float, default=0.0,
                        help="Fraction of each linear layer's weights to zero via global "
                             "L1-unstructured magnitude pruning, after merging. 0.0 (default) "
                             "disables it — see prune_model()'s docstring for why it's opt-in.")
    parser.add_argument("--qlora", action="store_true",
                        help="Load the base model 4-bit (bitsandbytes) for QLoRA instead of the "
                             "default full-precision LoRA. Needs a CUDA GPU — refuses cleanly "
                             "without one. See load_base_model()'s docstring.")
    parser.add_argument("--auto-hyperparams", action="store_true",
                        help="Let the PEFT hyperparameter bandit (peft_hyperparam_optimizer.py) "
                             "pick --lora-r/--lora-alpha/--min-reward for tonight, learned from "
                             "past regression-gate outcomes, instead of using the fixed values "
                             "above. Off by default — opt in once regression_gate.py is also "
                             "wired with --update-peft-bandit to close the feedback loop, "
                             "otherwise the bandit never learns anything and just adds noise.")
    parser.add_argument("--peft-bandit-state", default="data/finetune/peft_bandit_state.json",
                        help="Where the hyperparameter bandit's learned posteriors persist "
                             "across nights (see peft_hyperparam_optimizer.PeftHyperparamOptimizer).")
    args = parser.parse_args()

    if args.auto_hyperparams:
        import peft_hyperparam_optimizer as peft_bandit
        all_raw = load_examples(Path(args.log_path), min_reward=0.0, max_examples=10_000)
        avg_reward = sum(e.get("reward", 0.0) for e in all_raw) / len(all_raw) if all_raw else 0.0
        bandit = peft_bandit.PeftHyperparamOptimizer(args.peft_bandit_state)
        arm, hp, context = bandit.select_arm_full(curated_count=len(all_raw), avg_reward=avg_reward)
        args.lora_r, args.lora_alpha, args.min_reward = hp["lora_r"], hp["lora_alpha"], hp["min_reward"]
        print(f"--auto-hyperparams: bandit picked arm={arm!r} -> {hp} "
             f"(context: {len(all_raw)} raw examples, avg reward {avg_reward:.2f})")
        # Written for regression_gate.py's --update-peft-bandit to read once
        # tonight's fine-tune has actually been scored — this process exits
        # long before that happens, so the arm/context has to survive on disk.
        sidecar = Path(args.output_dir) / "peft_arm_used.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"arm": arm, "context": context,
                                       "state_path": args.peft_bandit_state}), encoding="utf-8")

    raw_examples = load_examples(Path(args.log_path), min_reward=args.min_reward,
                                 max_examples=args.max_examples)
    if args.no_data_engine:
        examples = raw_examples
    else:
        store = data_engine.DataEngineStore(args.curated_path, dedup_threshold=args.dedup_threshold)
        store.ingest(raw_examples)  # scrubs + persists only what's genuinely new tonight
        examples = store.load_all()[-args.max_examples:]  # train on the curated corpus, most-recent-capped

    if len(examples) < args.min_examples:
        print(f"Only {len(examples)} curated example(s) meet reward >= {args.min_reward} "
             f"(need >= {args.min_examples}) — nothing to train on tonight, exiting cleanly.")
        return

    print(f"Training on {len(examples)} example(s)"
         f"{' from ' + args.log_path if args.no_data_engine else ' from the curated store ' + args.curated_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(args.base_model, qlora=args.qlora)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(examples, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        # Trainer's default column-pruning drops "reward" before collation
        # since it's not a model.forward() arg — it isn't unused, our
        # custom compute_loss() needs it.
        remove_unused_columns=False,
    )
    trainer = RewardWeightedTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[WallClockStop(args.max_minutes)],
    )
    trainer.train()

    merged = model.merge_and_unload()
    if args.prune_sparsity > 0.0:
        actual_sparsity = prune_model(merged, args.prune_sparsity)
        print(f"Pruned to {actual_sparsity:.1%} sparsity across linear layers "
             f"(requested {args.prune_sparsity:.1%}) — see prune_model()'s docstring "
             f"for the honest limitation on whether this helps the GGUF serving path.")
    merged_dir = Path(args.output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir} — "
         f"next: scripts/convert_and_register_ollama_model.sh {merged_dir}")


if __name__ == "__main__":
    main()

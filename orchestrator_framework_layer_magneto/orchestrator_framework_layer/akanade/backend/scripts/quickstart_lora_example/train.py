"""Quickstart LoRA training example — a minimal, single-file way to see
the SLM Generation Layer's core idea (take an open base model, train a
LoRA adapter on your own data) work end to end on a laptop GPU or a
free Colab/Kaggle notebook, in minutes rather than a full nightly run.

Deliberately standalone (own requirements.txt, no import of the `app`
package or of scripts/finetune_local_slm.py) — same isolation
convention as every other scripts/ entry point in this pipeline, since
torch/peft/trl/bitsandbytes pull in a heavy, GPU-specific dependency
set that has no business inside the always-running chatbot image.

This is NOT the production nightly pipeline (that's
scripts/finetune_local_slm.py + scripts/data_engine.py +
scripts/convert_and_register_ollama_model.sh, run in Docker, reward-
weighted on real logged chat turns, gated by
scripts/eval/regression_gate.py before anything reaches a real user).
This script trains on the tiny hand-written data/train.jsonl below and
never touches Ollama or the live chatbot — it's for learning the shape
of the pipeline and iterating on a data format fast. To turn a real
result from here into an actual router-selectable model, merge the
adapter (see inference.py) into a full HF model directory the same way
finetune_local_slm.py's own `merge_and_unload()` does, then hand that
directory to convert_and_register_ollama_model.sh.
"""
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B"

DATASET_PATH = "data/train.jsonl"

OUTPUT_DIR = "./algo-akanade-slm-quickstart"


# --------------------------------------------------
# 1. Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# 2. Quantization
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# --------------------------------------------------
# 3. Load Base Model
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False


# --------------------------------------------------
# 4. Load Dataset
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)

dataset = dataset["train"]


# --------------------------------------------------
# 5. Convert examples to chat format
# --------------------------------------------------

def format_example(example):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Algo-Akanade-SLM, an enterprise "
                "intent classification model."
            )
        },
        {
            "role": "user",
            "content": (
                example["instruction"]
                + "\n\n"
                + example["input"]
            )
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }


dataset = dataset.map(
    format_example
)


# --------------------------------------------------
# 6. LoRA configuration
# --------------------------------------------------

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)


# --------------------------------------------------
# 7. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=200,

    save_total_limit=2,

    fp16=True,

    optim="paged_adamw_8bit",

    report_to="none",

)


# --------------------------------------------------
# 8. Trainer
# --------------------------------------------------

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config,

    args=training_args,

)


# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("Starting Algo-Akanade-SLM quickstart training...")

trainer.train()


# --------------------------------------------------
# 10. Save Adapter
# --------------------------------------------------

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("Training complete.")

print(
    f"LoRA adapter saved to: {OUTPUT_DIR}"
)

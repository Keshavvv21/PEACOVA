"""Loads the LoRA adapter train.py produced and runs one prompt through
it, base model + adapter, no merge — just to see the quickstart result
answer before deciding whether it's worth carrying further into the
real pipeline (merge_and_unload() -> convert_and_register_ollama_model.sh).
"""
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-0.6B"

ADAPTER = "./algo-akanade-slm-quickstart"


# -----------------------------------------
# Load tokenizer
# -----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER
)


# -----------------------------------------
# Load base model
# -----------------------------------------

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    torch_dtype=torch.float16,

    device_map="auto",

)


# -----------------------------------------
# Load your LoRA adapter
# -----------------------------------------

model = PeftModel.from_pretrained(
    model,
    ADAPTER
)

model.eval()


# -----------------------------------------
# User request
# -----------------------------------------

query = """
Classify this request:

Summarize the customer's contract
and identify financial risks.
"""


messages = [

    {
        "role": "system",
        "content": (
            "You are Algo-Akanade-SLM, "
            "an enterprise intent classifier."
        ),
    },

    {
        "role": "user",
        "content": query,
    }

]


# -----------------------------------------
# Tokenize
# -----------------------------------------

inputs = tokenizer.apply_chat_template(

    messages,

    add_generation_prompt=True,

    return_tensors="pt"

).to(model.device)


# -----------------------------------------
# Generate
# -----------------------------------------

with torch.no_grad():

    outputs = model.generate(

        inputs,

        max_new_tokens=150,

        temperature=0.1,

        do_sample=False,

    )


# -----------------------------------------
# Decode
# -----------------------------------------

response = tokenizer.decode(

    outputs[0][inputs.shape[-1]:],

    skip_special_tokens=True

)


print("\nAlgo-Akanade-SLM (quickstart):")
print(response)

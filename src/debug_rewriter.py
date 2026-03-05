"""
Run this standalone to see exactly what DeepSeek-1.3B is generating.
Usage: poetry run python src/debug_rewriter.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import settings

TEST_CODE = """
def candy(ratings):
    candies = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    for i in range(len(ratings)-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)
""".strip()

print(f"Loading {settings.REWRITER_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(settings.REWRITER_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    settings.REWRITER_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = model.to(settings.DEVICE)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Try a few different prompt styles and print raw output ────────────────────

prompts = {
    "alpaca_style": (
        f"### Instruction:\n"
        f"Please explain the functionality of the given code, "
        f"then rewrite it in a single markdown code block. "
        f"No additional clarifications.\n\n"
        f"### Code:\n{TEST_CODE}\n\n"
        f"### Response:\n"
    ),
    "simple_chat": (
        f"Rewrite the following Python code in a single markdown code block:\n\n"
        f"```python\n{TEST_CODE}\n```\n\n"
        f"Rewritten code:\n"
    ),
    "deepseek_chat_template": None,  # uses apply_chat_template if available
}

# Also try the official chat template if the tokenizer supports it
if hasattr(tokenizer, "apply_chat_template"):
    messages = [
        {
            "role": "user",
            "content": (
                f"Please explain the functionality of the given code, "
                f"then rewrite it in a single markdown code block.\n\n"
                f"```python\n{TEST_CODE}\n```"
            )
        }
    ]
    try:
        prompts["deepseek_chat_template"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Chat template failed: {e}")
        prompts.pop("deepseek_chat_template")

for name, prompt in prompts.items():
    if prompt is None:
        continue

    print(f"\n{'='*60}")
    print(f"PROMPT STYLE: {name}")
    print(f"{'='*60}")
    print("--- PROMPT ---")
    print(prompt)
    print("--- RAW OUTPUT ---")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(settings.DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs.input_ids.shape[1]
    generated = output[0][prompt_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    print(decoded)
    print(f"\n[Length: {len(decoded)} chars]")
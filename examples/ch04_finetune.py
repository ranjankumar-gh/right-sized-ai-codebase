# SYNTHETIC INPUTS: data/task.jsonl and fixtures/serving_render.txt are
# hand-written placeholders a few lines long. The script does not run end
# to end as shipped: it prints the trainable count, then stops at guard two,
# the check_template_match call on line 32, which raises TemplateDrift by
# design. Replace both placeholders with real data and a real serving-path
# capture before any training you intend to keep.
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from rightsize.adapt import (
    LORA_TARGETS_ALL,
    adapter_parameter_count,
    check_loss_mask,
    check_template_match,
)

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
RANK = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(MODEL_ID).to_dict()
print(f"trainable: {adapter_parameter_count(config, RANK):,}")

# Guard two, before any GPU time is spent. serving_render must come from
# the code path production uses, not from this tokenizer, or the check
# is comparing a helper against itself.
sample = [
    {"role": "user", "content": "Summarize the incident in one sentence."},
]
check_template_match(
    training_render=tokenizer.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True
    ),
    serving_render=open("fixtures/serving_render.txt", encoding="utf-8").read(),
)

peft_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK * 2,
    lora_dropout=0.05,
    target_modules=list(LORA_TARGETS_ALL),
    task_type="CAUSAL_LM",
)

config = SFTConfig(
    output_dir="out/smollm3-task",
    # Conversational data, so this is the masking switch that applies.
    # completion_only_loss is supported only for prompt-completion datasets,
    # so it does not apply to this conversational data.
    assistant_only_loss=True,
    max_length=2048,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    # Trades recompute for activation memory. Turn it off only after
    # estimate_training_memory says the activations fit without it.
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=MODEL_ID,
    args=config,
    train_dataset=load_dataset("json", data_files="data/task.jsonl")["train"],
    peft_config=peft_config,
)

# Guard one, on a real batch from the real collator.
batch = next(iter(trainer.get_train_dataloader()))
fraction = check_loss_mask(batch["labels"][0].tolist())
print(f"supervised fraction: {fraction:.1%}")

trainer.train()

# The adapter and tokenizer that examples/ch04_preference.py loads.
trainer.save_model("out/smollm3-task")

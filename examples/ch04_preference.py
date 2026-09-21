# SYNTHETIC INPUT: data/preferences.jsonl is a hand-written placeholder a few
# lines long. Replace it with real preference pairs before real training.
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from rightsize.adapt import LORA_TARGETS_ALL

# Exactly three columns, and only the final assistant turn may differ
# between chosen and rejected. A pair that differs earlier trains on
# the difference you did not mean to express.
pairs = load_dataset("json", data_files="data/preferences.jsonl")["train"]
assert set(pairs.column_names) == {"prompt", "chosen", "rejected"}

config = DPOConfig(
    output_dir="out/smollm3-task-dpo",
    # Preference strength. Below roughly 0.05 the signal is ignored;
    # above 1.0 the run tends to spike. Start here and move it only
    # after reading the reward-margin curve.
    beta=0.1,
    # Lower than the SFT stage: this runs on top of an already-adapted
    # model and is correcting rather than teaching.
    learning_rate=5e-6,
    # max_length bounds the whole sequence, prompt included. DPOConfig has
    # no separate max_prompt_length in current TRL.
    max_length=2048,
    per_device_train_batch_size=2,
    bf16=True,
)

trainer = DPOTrainer(
    model="out/smollm3-task",  # the SFT result, not the base model
    args=config,
    train_dataset=pairs,
    peft_config=LoraConfig(r=16, lora_alpha=32,
                           target_modules=list(LORA_TARGETS_ALL),
                           task_type="CAUSAL_LM"),
)
trainer.train()

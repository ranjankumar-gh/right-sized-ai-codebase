from rightsize.capability import parameter_budget, read_candidate

# Values read from google/gemma-3-270m config.json.
gemma_270m = {
    "vocab_size": 262144,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "num_hidden_layers": 18,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "tie_word_embeddings": True,
}

budget = parameter_budget(gemma_270m)
print(f"embedding:     {budget.embedding / 1e6:.1f}M")
print(f"transformer:   {budget.non_embedding / 1e6:.1f}M")
print(f"total:         {budget.total / 1e6:.1f}M")
print(f"embedding pct: {budget.embedding_fraction:.0%}")


smollm3 = {
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "tie_word_embeddings": True,
}

budget = parameter_budget(smollm3)
compute = budget.non_embedding
print(f"attention: {budget.attention / 1e6:8.1f}M  {budget.attention / compute:.0%}")
print(f"mlp:       {budget.mlp / 1e6:8.1f}M  {budget.mlp / compute:.0%}")


shortlist = [
    read_candidate("Gemma 3 270M", gemma_270m, 0.27, 6.0),
    read_candidate("SmolLM3 3B", smollm3, 3.08, 11.2),
]

print(f"{'model':<16}{'multiple':>12}{'compute params':>16}{'tax':>7}")
for candidate in shortlist:
    print(candidate.summary_row())

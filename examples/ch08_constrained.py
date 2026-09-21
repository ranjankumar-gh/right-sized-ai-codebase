from rightsize.structured import allowed_continuations, mask_logits, pick

DECISIONS = ["APPROVE", "REJECT", "ESCALATE"]

# What the model actually believed on a genuinely ambiguous case: it
# wanted to escalate, and its second choice was a coin flip.
beliefs = {"E": 0.61, "A": 0.20, "R": 0.19}

allowed = allowed_continuations("", DECISIONS)
print(f"legal first characters: {sorted(allowed)}")
print(f"unconstrained choice:   {pick(beliefs)}")
print(f"constrained choice:     {pick(mask_logits(beliefs, allowed))}")

# Now the same model, on a schema that dropped ESCALATE because the
# product team simplified it to a binary.
binary = ["APPROVE", "REJECT"]
allowed = allowed_continuations("", binary)
forced = pick(mask_logits(beliefs, allowed))
print(f"\nwith ESCALATE removed from the schema:")
print(f"legal first characters: {sorted(allowed)}")
print(f"constrained choice:     {forced}")
print(f"model's confidence in it: {beliefs[forced]:.0%}")


from rightsize.structured import schema_complexity

flat = {
    "properties": {
        "decision": {"enum": ["APPROVE", "REJECT", "ESCALATE"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
}

nested = {
    "properties": {
        "result": {
            "properties": {
                "decision": {"enum": ["APPROVE", "REJECT", "ESCALATE"]},
                "evidence": {
                    "properties": {
                        "source": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["source", "confidence"],
                },
            },
            "required": ["decision"],
        },
    },
    "required": ["result"],
}

for name, schema in (("flat", flat), ("nested", nested)):
    c = schema_complexity(schema)
    print(
        f"{name:7} depth={c.max_depth} fields={c.field_count} "
        f"required={c.required_count} enums={c.enum_branches} "
        f"pressure={c.pressure_score:.1f}"
    )

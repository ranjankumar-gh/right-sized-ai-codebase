from dataclasses import dataclass

# Hoffmann et al. (2022): for a fixed TRAINING compute budget, loss is
# minimized near this ratio. It is a budget-allocation result, not a
# quality target, and it is the denominator here for exactly that reason.
CHINCHILLA_TOKENS_PER_PARAM = 20.0


@dataclass(frozen=True)
class TrainingProfile:
    """Two numbers off a model card, and what they imply."""

    name: str
    total_params_b: float
    train_tokens_t: float

    @property
    def tokens_per_param(self) -> float:
        return (self.train_tokens_t * 1e12) / (self.total_params_b * 1e9)

    @property
    def overtraining_multiple(self) -> float:
        """How many times past compute-optimal this model was trained."""
        return self.tokens_per_param / CHINCHILLA_TOKENS_PER_PARAM


@dataclass(frozen=True)
class ParameterBudget:
    """Where a model's parameters live, counted from its config."""

    embedding: int
    attention: int
    mlp: int
    norm: int

    @property
    def total(self) -> int:
        return self.embedding + self.attention + self.mlp + self.norm

    @property
    def non_embedding(self) -> int:
        """The parameters that do the computing."""
        return self.attention + self.mlp + self.norm

    @property
    def embedding_fraction(self) -> float:
        return self.embedding / self.total


def parameter_budget(config: dict, gated_mlp: bool = True) -> ParameterBudget:
    """Count parameters by role from a Hugging Face config.json.

    gated_mlp is True for every current family, which uses a three-matrix
    gated feed-forward block (gate, up, down). Models predating that use
    two. The config does not reliably say which: Gemma reports a plain
    activation name and still uses a gated block, so this is a flag and
    not a sniff.
    """
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    inter = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim") or hidden // heads

    # An untied model stores the output projection separately, doubling
    # the table. Tying is near-universal below 4B for that reason.
    tables = 1 if config.get("tie_word_embeddings", False) else 2
    embedding = config["vocab_size"] * hidden * tables

    q_and_o = 2 * hidden * heads * head_dim
    k_and_v = 2 * hidden * kv_heads * head_dim
    attention = (q_and_o + k_and_v) * layers

    mlp = (3 if gated_mlp else 2) * hidden * inter * layers
    norm = 2 * hidden * layers  # Approximate; norms are under 0.1%.

    return ParameterBudget(embedding, attention, mlp, norm)


@dataclass(frozen=True)
class Candidate:
    """A model described entirely by public information."""

    profile: TrainingProfile
    budget: ParameterBudget

    @property
    def name(self) -> str:
        return self.profile.name

    def summary_row(self) -> str:
        return (
            f"{self.name:<16}"
            f"{self.profile.overtraining_multiple:>11,.0f}x"
            f"{self.budget.non_embedding / 1e9:>15.2f}B"
            f"{self.budget.embedding_fraction:>7.0%}"
        )


def read_candidate(
    name: str, config: dict, total_params_b: float, train_tokens_t: float
) -> Candidate:
    """Everything derivable about a model before you download it."""
    return Candidate(
        profile=TrainingProfile(name, total_params_b, train_tokens_t),
        budget=parameter_budget(config),
    )

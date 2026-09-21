import torch
import torch.nn.functional as F


def batched_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """The same objective over a batch. Shapes: (batch, sequence, vocab)."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * temperature**2
    )

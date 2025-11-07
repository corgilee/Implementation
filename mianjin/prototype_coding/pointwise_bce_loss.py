import torch
import torch.nn.functional as F


def pointwise_bce_loss(scores: torch.Tensor,
                       labels: torch.Tensor,
                       slate_ids: torch.Tensor) -> torch.Tensor:
    """
    Pointwise BCE loss for multi-slate ranking with implicit feedback.

    Args:
        scores:    [N] raw model logits for each (slate, item).
        labels:    [N] binary target (clicked=1, skipped=0, etc.).
        slate_ids: [N] grouping ids (not used in the loss itself but kept
                    to highlight we are handling multiple slates).

    Returns:
        Scalar tensor with the average binary cross-entropy loss.
    """

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    loss = F.binary_cross_entropy_with_logits(scores, labels.float())
    return loss


if __name__ == "__main__":
    slate_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    labels = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
    scores = torch.randn_like(labels)

    print("Pointwise BCE loss:", pointwise_bce_loss(scores, labels, slate_ids).item())

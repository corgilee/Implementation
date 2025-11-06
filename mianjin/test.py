import os
import sys

import torch


CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "prototype_coding"))

from listwise_loss import listwise_softmax_ranking_loss  # noqa: E402


def _manual_listwise_loss(pos_scores: torch.Tensor,
                          neg_scores: torch.Tensor) -> torch.Tensor:
    logits = torch.cat([pos_scores.view(-1, 1), neg_scores], dim=1)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return (-log_probs[:, 0]).mean()


def test_matches_manual_computation():
    pos = torch.tensor([3.0, 1.5])
    neg = torch.tensor([[1.0, 0.5], [0.2, -0.3]])

    expected = _manual_listwise_loss(pos, neg)
    actual = listwise_softmax_ranking_loss(pos, neg)

    torch.testing.assert_close(actual, expected)


def test_loss_decreases_when_positive_improves():
    pos_low = torch.tensor([0.5])
    pos_high = torch.tensor([1.5])
    neg = torch.tensor([[1.0, 0.2]])

    loss_low = listwise_softmax_ranking_loss(pos_low, neg)
    loss_high = listwise_softmax_ranking_loss(pos_high, neg)

    assert loss_high < loss_low


if __name__ == "__main__":
    test_matches_manual_computation()
    test_loss_decreases_when_positive_improves()
    print("All tests passed.")

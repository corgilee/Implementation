import torch
import torch.nn.functional as F

print("Implement pairwise loss")



def pairwise_logistic_loss(pos_scores: torch.Tensor,
                      neg_scores: torch.Tensor,
                      mask: torch.Tensor | None = None,
                      reduction: str = "mean") -> torch.Tensor:
    pos = pos_scores.view(-1, 1)          # [B,1]
    delta = pos - neg_scores              # [B,K]
    loss = F.softplus(-delta)             # log(1+exp(-Δ))
    if mask is not None:
        loss = loss * mask
        if reduction == "mean":
            denom = mask.sum().clamp_min(1)
            return loss.sum() / denom
    return loss.mean() if reduction == "mean" else loss.sum()


B, K = 4, 3
# pos = torch.tensor([2.0, 1.2, 0.5, 3.0])        # 正样本分数 [B]
# negs = torch.tensor([[1.5, 0.2, -0.4],          # 负样本分数 [B, K]
#                      [0.9, 0.7,  0.1],
#                      [0.6, 0.4, -0.1],
#                      [2.2, 1.0,  0.0]])

# base random uniform in [0,1]
pos  = torch.rand(B) * 0.5 + 0.5      # [0.5 ~ 1.0]
negs = torch.rand(B, K) * 0.5         # [0.0 ~ 0.5]

print(pos)
print(negs)

#print("hinge  :", pairwise_hinge_loss(pos, negs, margin=1.0).item())
print("bpr    :", pairwise_logistic_loss(pos, negs).item())

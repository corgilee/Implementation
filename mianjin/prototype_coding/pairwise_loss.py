import torch
import torch.nn.functional as F

print("Implement pairwise loss")



def pairwise_logistic_loss(pos, negs):
    pos= pos.view(-1,1) #add 1 more dimension
    print(pos.shape)
    print(negs.shape)
    delta = pos - negs
    print(f'dela_shape: {delta.shape}')
    
    loss=F.softplus(-delta).mean()
    print(f'loss, {loss}')
    return loss



def listwise_softmax_ce(pos, negs):
    """
    pos:  [B]
    negs: [B,K]
    return scalar
    """
    # concat so positive is column 0:  [B, 1+K]
    scores = torch.cat([pos.view(-1,1), negs], dim=1)   # [B, K+1]
    print(scores)
    
    # correct class index (always 0, because we put pos at column 0)
    target = torch.zeros(scores.size(0), dtype=torch.long)
    print(target)
    
    # standard cross-entropy
    loss = F.cross_entropy(scores, target)
    return loss

B, K = 4, 3

#help(torch.rand)
pos  = torch.rand(B) * 0.5 + 0.5      # [0.5 ~ 1.0]
negs = torch.rand(B, K) * 0.5         # [0.0 ~ 0.5]

print(pos)
print(negs)

#print("hinge  :", pairwise_hinge_loss(pos, negs, margin=1.0).item())
'''
For two-tower retrieval with implicit feedback, I default to BPR — it’s smooth, robust, no margin tuning. 
BPR is O(B×K) because each positive score is compared with K negatives.

Listwise CE I use more in re-ranking where I have a full candidate list.
Listwise CE is O(B×(K+1)) because each anchor runs a softmax over its K+1 candidates.

'''

print("bpr    :", pairwise_logistic_loss(pos, negs).item())
print("listwise loss:", listwise_softmax_ce(pos, negs).item())

'''
listwise CE treats the positive + negatives as a group, runs softmax across them, and maximizes the probability of the positive being ranked highest.
'''



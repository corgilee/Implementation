import torch
import torch.nn.functional as F

def listnet_loss(scores, labels, slate_ids):
    """
    scores:    Tensor [N] - model outputs for each (slate, item)
    labels:    Tensor [N] - true relevance scores (e.g. clicked=1, skip=0, CVR label, etc.)
    slate_ids: Tensor [N] - identifier to group items that belong to the same slate/request
    """
    total_loss = 0.0
    num_slates = 0

    # group by slate
    sorted_idx = torch.argsort(slate_ids)       # ensure items from same slate are contiguous
    slate_ids = slate_ids[sorted_idx]           # reorder slate ids accordingly
    scores    = scores[sorted_idx]
    labels    = labels[sorted_idx]

    # loop every slate
    
    for sid in slate_ids.unique():
        mask = (slate_ids == sid)
        print(mask)

        s = scores[mask]   # predicted scores for this slate
        y = labels[mask]   # true relevance labels for this slate
        print(s,y)

        # softmax normalization inside slate
        P = F.softmax(y, dim=0)   # target distribution
        Q = F.softmax(s, dim=0)   # predicted distribution

        # cross entropy(P || Q)
        loss_slate = -(P * torch.log(Q)).sum()

        total_loss += loss_slate
        num_slates += 1

    return total_loss / num_slates


##### test case ########
slate_ids = torch.tensor([0,0,0, 1,1,1,])

labels = torch.rand(6)      # random relevance
scores = torch.rand(6)      # random model outputs

loss = listnet_loss(scores, labels, slate_ids)
print("ListNet loss:", loss.item())


'''
1. implement a retrival algorithm for user_query search
2. implement a loss function for ranking



'''
import torch
import torch.nn.functional as F


# ##########################
# # pipeline demo
# ##########################
query = "wireless noise cancelling headphones"
documents = [
    "bluetooth earbuds with long battery",
    "wireless over-ear noise cancelling headset",
    "tv remote control batteries",
    "gaming keyboard with rgb lights",
    "apple airpods pro 2",
    "headphones"
]


# for index,val in enumerate(documents):
#     print(f"index:{index}, val:{val}")

# retrieval methodology , 
# 1. overlapped_token 2. calculated embedding
def retrieve_candidates(query, documents,k=2):
    # count the overlapped tokens to count
    query_token=set(query.lower().split())
    overlap_list=[]
    for index, doc in enumerate(documents):
        doc_token=set(doc.lower().split())
        score=len(set(query_token & doc_token))/len(set(query_token))
        overlap_list.append((index,score))
        
    overlap_list.sort(key=lambda x: x[1], reverse=True)
    
    res=[]
    for i in range(k):
        res.append(overlap_list[i][0])

    return res
    

# #################################
# # embedding-based retrieval demo
# #################################

import torch.nn.functional as F

torch.manual_seed(0)
EMB_DIM = 8

# pretend we already have embeddings for the query/documents
query_embedding = torch.randn(EMB_DIM)
document_embeddings = torch.randn(len(documents), EMB_DIM)


def retrieve_candidates_embedding(query_emb: torch.Tensor,
                                  doc_embs: torch.Tensor,
                                  k: int = 2):
    """Return top-k document indices by cosine similarity."""
    query_emb = F.normalize(query_emb, dim=0)
    doc_embs = F.normalize(doc_embs, dim=1)

    scores = torch.matmul(doc_embs, query_emb)  # cosine similarity because vectors normalized
    topk = torch.topk(scores, k).indices.tolist()
    return topk

# # retrieval step
cands = retrieve_candidates(query, documents, k=2)
embedding_cands = retrieve_candidates_embedding(query_embedding, document_embeddings, k=2)

print("token overlap:", cands)
print("embedding    :", embedding_cands)

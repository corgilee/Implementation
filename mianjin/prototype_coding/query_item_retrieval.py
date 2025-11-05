
'''
1. implement a retrival algorithm for user_query search
2. implement a loss function for ranking



'''
import torch


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

# retrieval methodology , 1. overlapped_token 2. calculated embedding
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
    

# from sentence_transformers import SentenceTransformer, util

# # load model once (outside function)
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def retrieve_candidates_embedding(query, documents, k=2):
#     # embed
#     q_emb  = embed_model.encode([query])
#     d_embs = embed_model.encode(documents)

#     # cosine similarity [1, len(documents)]
#     scores = util.cos_sim(q_emb, d_embs).squeeze(0)   # -> tensor [D]

#     # topk
#     topk = torch.topk(scores, k).indices.tolist()
#     return topk

# # retrieval step
cands = retrieve_candidates(query, documents, k=2)

print(cands)


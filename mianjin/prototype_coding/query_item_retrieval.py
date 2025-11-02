
'''
1. implement a retrival algorithm for user_query search
2. implement a loss function for ranking

'''


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
    


# # retrieval step
cands = retrieve_candidates(query, documents, k=2)

print(cands)

# # suppose index0 is positive, index1 is negative
# pos = cands[0]
# neg = cands[1]

# pos_s = score(query, pos)
# neg_s = score(query, neg)

# print("pos:", pos, pos_s)
# print("neg:", neg, neg_s)
# print("loss:", ranking_loss(pos_s, neg_s))
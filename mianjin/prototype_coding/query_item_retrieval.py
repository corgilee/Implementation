
'''
1. implement a retrival algorithm for user_query search
2. implement a loss function for ranking

'''






# ##########################
# # pipeline demo
# ##########################
# query = "wireless noise cancelling headphones"
# documents = [
#     "bluetooth earbuds with long battery",
#     "wireless over-ear noise cancelling headset",
#     "tv remote control batteries",
#     "gaming keyboard with rgb lights",
#     "apple airpods pro 2"
# ]

# # retrieval step
# cands = retrieve_candidates(query, documents, k=2)

# # suppose index0 is positive, index1 is negative
# pos = cands[0]
# neg = cands[1]

# pos_s = score(query, pos)
# neg_s = score(query, neg)

# print("pos:", pos, pos_s)
# print("neg:", neg, neg_s)
# print("loss:", ranking_loss(pos_s, neg_s))
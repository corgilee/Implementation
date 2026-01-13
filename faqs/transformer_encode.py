import math
import numpy as np

def transformer_encoder_layer(X, W_Q, W_K, W_V, W_O, ffn, layer_norm, num_heads):
    """
    X: (n_tokens, d_model)
    """

    # ---- Self-attention ----
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[1] // num_heads

    def split_heads(M):
        return np.split(M, num_heads, axis=1)

    Q_heads = split_heads(Q)
    K_heads = split_heads(K)
    V_heads = split_heads(V)

    head_outputs = []

    for Qi, Ki, Vi in zip(Q_heads, K_heads, V_heads):
        scores = (Qi @ Ki.T) / math.sqrt(d_k)
        weights = softmax(scores)
        head_outputs.append(weights @ Vi)

    # Concatenate heads
    attention_out = np.concatenate(head_outputs, axis=1)
    attention_out = attention_out @ W_O

    # ---- Residual + norm ----
    X1 = layer_norm(X + attention_out)

    # ---- Feed-forward ----
    ffn_out = ffn(X1)

    # ---- Residual + norm ----
    output = layer_norm(X1 + ffn_out)

    return output

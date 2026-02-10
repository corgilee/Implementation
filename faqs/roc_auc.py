import numpy as np

def calculate_roc_auc(y_true, y_scores):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. Sort scores and corresponding truth values in descending order
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]
    
    # 2. Calculate True Positives and False Positives at each threshold
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    # 3. Calculate TPR and FPR
    # Recall: TPR = TP / Total Positives; FPR = FP / Total Negatives
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    # 4. Add (0,0) to the start of the curve
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # 5. Use the Trapezoidal Rule to calculate Area Under the Curve
    # Area = sum of [ (x_i - x_{i-1}) * (y_i + y_{i-1}) / 2 ]
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)
    
    return auc


# Test Data
y_test = [0, 0, 1, 1]
y_probs = [0.1, 0.4, 0.35, 0.8]

# Calculate AUC
auc_score = calculate_roc_auc(y_test, y_probs)

print(f"True Labels:  {y_test}")
print(f"Model Probs:  {y_probs}")
print(f"ROC-AUC Score: {auc_score:.4f}")

# Verification with Scikit-learn (Optional comparison)
# from sklearn.metrics import roc_auc_score
# print(f"Sklearn Score: {roc_auc_score(y_test, y_probs):.4f}")

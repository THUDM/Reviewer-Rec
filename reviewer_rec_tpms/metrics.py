import math

def calculate_ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    idcg = 0.0
    
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            dcg += 1.0 / math.log(i + 2, 2)
            
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log(i + 2, 2)
        
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg

def calculate_recall_at_k(recommended, relevant, k):
    intersection = len(set(recommended[:k]) & set(relevant))
    return intersection / float(len(relevant))
def berechne_metriken(ground_truth: dict, predicted: dict):
    if not isinstance(ground_truth, dict): ground_truth = {}
    if not isinstance(predicted, dict): predicted = {}
    
    # Diese Kategorien werden bei der Metrik-Berechnung ignoriert
    ignored_keys = ["Krankheit"] 
    
    gt_set = set()
    for k, vals in ground_truth.items():
        if k in ignored_keys: continue  
        if isinstance(vals, list):
            for v in vals: gt_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")
            
    pred_set = set()
    for k, vals in predicted.items():
        if k in ignored_keys: continue 
        if isinstance(vals, list):
            for v in vals: pred_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")
            
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, tp, fp, fn
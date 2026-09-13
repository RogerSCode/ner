"""
Modul zur Berechnung quantitativer Evaluationsmetriken.
"""

def berechne_metriken(ground_truth: dict, predicted: dict):
    """
    Berechnet die quantitativen Metriken (Precision, Recall, F1-Score) für einen Text.

    Ignoriert spezifische Kategorien (z. B. 'Krankheit') und vergleicht
    die verbleibenden extrahierten Entitäten mit den Referenzdaten.

    Args:
        ground_truth (dict): Die manuell validierten Referenzdaten (Goldstandard).
        predicted (dict): Die von der KI extrahierten Entitäten.

    Returns:
        tuple: (precision, recall, f1_score, true_positives, false_positives, false_negatives)
    """
    if not isinstance(ground_truth, dict): ground_truth = {}
    if not isinstance(predicted, dict): predicted = {}
    
    # Diese Kategorien werden bei der Auswertung nicht berücksichtigt
    ignorierte_schluessel = ["Krankheit"] 
    
    referenz_set = set()
    for schluessel, werte in ground_truth.items():
        if schluessel in ignorierte_schluessel: continue  
        if isinstance(werte, list):
            for wert in werte: 
                referenz_set.add(f"{str(schluessel).lower()}:{str(wert).lower().strip()}")
            
    vorhersage_set = set()
    for schluessel, werte in predicted.items():
        if schluessel in ignorierte_schluessel: continue 
        if isinstance(werte, list):
            for wert in werte: 
                vorhersage_set.add(f"{str(schluessel).lower()}:{str(wert).lower().strip()}")
            
    echte_treffer = len(referenz_set.intersection(vorhersage_set)) # True Positives
    falscher_alarm = len(vorhersage_set - referenz_set) # False Positives
    uebersehen = len(referenz_set - vorhersage_set) # False Negatives
    
    precision = echte_treffer / (echte_treffer + falscher_alarm) if (echte_treffer + falscher_alarm) > 0 else 0.0
    recall = echte_treffer / (echte_treffer + uebersehen) if (echte_treffer + uebersehen) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score, echte_treffer, falscher_alarm, uebersehen
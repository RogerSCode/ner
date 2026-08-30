import json

def convert_gernermed_to_app_format(input_path: str, output_path: str):
    # 1. Lese die originale GERNERMED JSON-Datei ein
    with open(input_path, 'r', encoding='utf-8') as f:
        gernermed_data = json.load(f)
        
    extracted_data = []
    
    # 2. Übersetzungs-Wörterbuch (Englische GERNERMED-Typen zu unseren App-Keys)
    type_mapping = {
        "Drug": "Medikament",
        "Strength": "Wirkstaerke",
        "Form": "Form",
        "Dosage": "Dosierung",
        "Route": "Verabreichungsweg",
        "Frequency": "Haeufigkeit",
        "Duration": "Dauer"
    }
    
    # 3. Iteriere über alle Einträge im Datensatz
    for entry in gernermed_data:
        # Nimm den deutschen Text
        de_text = entry.get("de", "")
        annotations = entry.get("annotations", [])
        
        # Leeres Template für unsere App
        ground_truth = {
            "Krankheit": [], # Bleibt leer, da GERNERMED das nicht hat
            "Medikament": [],
            "Wirkstaerke": [],
            "Form": [],
            "Dosierung": [],
            "Verabreichungsweg": [],
            "Haeufigkeit": [],
            "Dauer": []
        }
        
        # Fülle das Template mit den gefundenen Annotationen
        for ann in annotations:
            ann_type = ann.get("type")
            content = ann.get("content")
            
            # Wenn der Typ in unserem Mapping ist, füge ihn hinzu
            if ann_type in type_mapping:
                target_key = type_mapping[ann_type]
                
                # Duplikate vermeiden
                if content not in ground_truth[target_key]:
                    ground_truth[target_key].append(content)
                    
        # Speichere das Ergebnis ab
        extracted_data.append({
            "text": de_text,
            "ground_truth": ground_truth
        })
            
    # 4. Exportiere das finale JSON für die App
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
    print(f"Erfolgreich {len(extracted_data)} Texte für die Streamlit App konvertiert!")

# Skript ausführen
if __name__ == "__main__":
    # Ändere hier den Dateinamen zu dem, wie du die Datei gespeichert hast
    convert_gernermed_to_app_format("GERNERMED_dataset.json", "app_testdaten.json")
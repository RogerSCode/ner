# 🤖 Agentic AI Medical NER System – Prototyp (Meilenstein 2)

Dieses Repository enthält den voll funktionsfähigen Prototypen (Meilenstein 2) einer agentenbasierten Named Entity Recognition (NER) Pipeline für medizinische Texte. Das System nutzt lokale Large Language Models (LLMs) via Ollama, um dem strengen Datenschutz im medizinischen Sektor gerecht zu werden, und verarbeitet klinische Rohtexte durch ein Multi-Agenten-System.

## 🌟 Kernfunktionen
* **Lokale Agenten-Architektur:** Ein Orchestrator-Agent wählt dynamisch die Prompt-Strategie (Zero-Shot, Few-Shot, Chain-of-Thought) basierend auf der Textkomplexität.
* **Self-Refinement:** Ein Critic-Agent prüft die anfängliche Extraktion kritisch auf Fehler oder Halluzinationen und korrigiert diese iterativ.
* **Datenschutz by Design:** Alle Inferenzschritte werden vollständig lokal über einen gekapselten Ollama-Docker-Container ausgeführt, ohne Daten an externe APIs (wie OpenAI) zu senden.
* **Interaktives Frontend:** Eine leichtgewichtige Streamlit-Oberfläche visualisiert den Agenten-Workflow und berechnet Evaluationsmetriken (Precision, Recall, F1-Score) in Echtzeit.

## 📂 Projektstruktur
* `app.py`: Das Streamlit-Frontend mit integrierter Evaluierungslogik.
* `agent_pipeline.py`: Beinhaltet die Logik des Orchestrator-, Extractor- und Critic-Agenten.
* `models.py`: Pydantic-Modelle zur typsicheren Interprozesskommunikation.
* `docker-compose.yaml` & `Dockerfile`: Orchestrieren die Container-Umgebung (GUI und lokaler LLM-Server).
* `requirements.txt`: Definiert die Python-Abhängigkeiten.

---

## 🚀 Installations- und Startanleitung

Voraussetzung für die Ausführung ist ein installiertes und laufendes **Docker** sowie **Docker Compose**. 

### Schritt 1: Container bauen und starten
Öffnen Sie ein Terminal im Hauptverzeichnis des Projekts (dort, wo die docker-compose.yaml liegt) und führen Sie folgenden Befehl aus:

    docker-compose up -d --build

*Dies startet zwei Container: Die Streamlit-App (ner_gui) und den lokalen LLM-Server (local_llm_agent_server).*

### Schritt 2: Sprachmodell in den Container laden (WICHTIG!)
Da der Ollama-Container beim ersten Start noch keine Modelle enthält, muss das verwendete Open-Source-Modell (Standard: llama3) einmalig in den Container geladen werden. 

Führen Sie dazu diesen Befehl im Terminal aus:

    docker exec -it local_llm_agent_server ollama run llama3

*Der Download (ca. 4.7 GB) startet. Sobald das Modell geladen ist, öffnet sich ein Chat-Prompt im Terminal. Beenden Sie diesen einfach durch Eingabe von /bye oder der Tastenkombination Strg + D.*

### Schritt 3: Benutzeroberfläche aufrufen
Die Anwendung ist nun vollständig lokal einsatzbereit. Öffnen Sie Ihren Webbrowser und navigieren Sie zu:
**👉 http://localhost:8501**

---

## 🛑 System beenden
Um die Container zu stoppen und die Ressourcen freizugeben, führen Sie im Projektverzeichnis folgenden Befehl aus:

    docker-compose down

*(Hinweis: Das heruntergeladene LLM-Modell bleibt durch ein gemountetes Volume persistent erhalten und muss beim nächsten Start nicht erneut heruntergeladen werden.)*

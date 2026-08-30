# 🤖 Agentic AI Medical NER System – Prototyp (Meilensteine 2 & 3)

Dieses Repository enthält den voll funktionsfähigen Prototypen einer agentenbasierten Named Entity Recognition (NER) Pipeline für medizinische Texte. Das System nutzt lokale Large Language Models (LLMs) via Ollama, um dem strengen Datenschutz im medizinischen Sektor gerecht zu werden, und verarbeitet klinische Rohtexte durch ein Multi-Agenten-System.

## 🌟 Kernfunktionen
* **Lokale Agenten-Architektur:** Ein Orchestrator-Agent wählt dynamisch die Prompt-Strategie (Zero-Shot, Few-Shot, Chain-of-Thought) basierend auf der Textkomplexität.
* **Self-Refinement:** Ein Critic-Agent prüft die anfängliche Extraktion kritisch auf Fehler oder Halluzinationen und korrigiert diese iterativ.
* **Datenschutz by Design:** Alle Inferenzschritte werden vollständig lokal über einen gekapselten Ollama-Docker-Container ausgeführt. Es fließen keine Daten an externe Cloud-Dienste.
* **Human-in-the-Loop & Persistenz:** Die GUI bietet einen interaktiven Dateneditor, in dem Fachexperten die KI-Extraktion validieren und als Goldstandard in einer lokalen Datenbank (`ground_truth_db.json`) speichern können.
* **Integrierte Batch-Evaluation:** Ein dedizierter Evaluierungs-Modus erlaubt das Hochladen von Testdatensätzen zur vollautomatischen Messung der Systemqualität (Precision, Recall, F1-Score).

## 📂 Projektstruktur
* `app.py`: Das Streamlit-Frontend mit zwei Anwendungsmodi (Klinisch & Evaluation).
* `agent_pipeline.py`: Beinhaltet die Logik des Orchestrator-, Extractor- und Critic-Agenten inkl. Pydantic-Validierung.
* `docker-compose.yaml` & `Dockerfile`: Orchestrieren die isolierte Container-Umgebung. Auf Code-Bind-Mounts wurde zur vollständigen Reproduzierbarkeit verzichtet.
* `requirements.txt`: Definiert die Python-Abhängigkeiten.
* `app_testdaten.json` *(optional)*: Konvertierter GERNERMED-Testdatensatz für die Batch-Evaluation.

---

## 🚀 Installations- und Startanleitung

Voraussetzung für die Ausführung ist eine installierte und laufende **Docker**-Umgebung.

### Schritt 1: Container bauen und starten
Öffnen Sie ein Terminal im Hauptverzeichnis des Projekts (dort, wo die `docker-compose.yaml` liegt) und führen Sie folgenden Befehl aus:

    docker compose up -d --build

*Dies startet die Streamlit-App und den lokalen LLM-Server in isolierten Containern.*

### Schritt 2: Benutzeroberfläche aufrufen
Die Anwendung ist nun einsatzbereit. Öffnen Sie Ihren Webbrowser und navigieren Sie zu:
**👉 http://localhost:8501**

### Schritt 3: Automatischer Modell-Download
Sie müssen LLM-Modelle nicht manuell vorladen! Wählen Sie einfach in der linken Seitenleiste der Streamlit-App das gewünschte Modell (z.B. `llama3` oder `qwen2:7b`) und starten Sie eine Analyse. 
*Hinweis: Fehlt das Modell im Ollama-Container, lädt die App es automatisch und transparent über einen Ladebalken in der GUI herunter.*

---

## 🛑 System beenden
Um die Container zu stoppen und die Ressourcen freizugeben, führen Sie im Projektverzeichnis folgenden Befehl aus:

    docker compose down

*(Hinweis: Heruntergeladene LLM-Modelle sowie die Datenbank `ground_truth_db.json` bleiben durch persistente Docker-Volumes erhalten und stehen beim nächsten Start sofort wieder zur Verfügung.)*
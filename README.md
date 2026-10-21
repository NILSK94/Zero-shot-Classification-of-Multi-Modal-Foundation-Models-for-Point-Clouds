# Multimodal Point Cloud Classifier

MMFM Lab erzeugt isometrische Punktwolkenansichten, klassifiziert Objekte mit multimodalen Modellen und wertet gespeicherte Ergebnisse aus.

## Starten

Python 3.12 verwenden. Unter Windows `START_TOOL.bat` starten. Alternativ:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Für die Installation unter Windows die virtuelle Umgebung zuerst mit `.venv\Scripts\activate` aktivieren; unter Linux/macOS mit `source .venv/bin/activate`. Der Windows-Starter übernimmt Einrichtung und Start automatisch.

## Bedienung

In der Seitenleiste die Datenordner und gewünschten Modelle auswählen. Zugangsdaten können dort eingegeben oder über die Variablen aus `.env.example` gesetzt werden. Der Mock-Modus prüft den Ablauf ohne Modellanfragen.

- **Übersicht:** verfügbare Objekte, Klassen und Versuchsbedingungen prüfen.
- **S3DIS vorbereiten:** Ansichten aus den Punktwolken erstellen.
- **JEV Bildtest:** ein Objekt mit einer oder zwei Ansichten testen; Anfrage und Kriterien vor dem kostenpflichtigen Aufruf prüfen.
- **Experimente:** ModelNet mit einer bis vier Ansichten und S3DIS mit RGB bzw. RGB+AABB ausführen. Jede Antwort wird sofort gespeichert.
- **Ergebnisse:** gespeicherte Läufe, automatische Zusammenfassungen, historische Ergebnisse und den optionalen Matrix-Explorer ansehen.
- **Auswertungen:** Grafiken und Daten aus gespeicherten Ergebnissen erzeugen und herunterladen. Dabei entstehen keine neuen Modellanfragen.

## Auswertungen

Ergebnisordner, Wiederholung und Ausgabeordner auswählen und **Auswertungen erzeugen** anklicken. ModelNet- und S3DIS-Übersichten, vollständige und fokussierte Konfusionsmatrizen, Jev-Wahrscheinlichkeiten und optional die historische Pilot-Auswertung werden automatisch erstellt.

Für S3DIS entstehen zusätzlich `S3DIS_Klassenvergleich_RGB_RGB_AABB.png` und `S3DIS_AABB_gepaarte_Klassifikationswechsel.png`. Der Klassenvergleich zeigt die richtig klassifizierten Objekte je Klasse für beide Eingaben. Die Wechselgrafik trennt falsch → richtig und richtig → falsch. Die Paarung erfolgt über Objekt-ID und Ground Truth. Jeder Prozentwert verwendet alle Objekte der entsprechenden wahren Klasse als Nenner. Alle vier Richtigkeitszustände und die Klassengrößen stehen in den exportierten Daten.

Die Oberfläche bietet einzelne PNGs und PDFs unter gleichbleibenden Dateinamen an. PNGs werden mit 600 dpi erzeugt. `MMFM_Auswertungen.zip` enthält ausschließlich Grafiken, Daten und die Herkunftsinformationen; keine Manuskriptanleitungen oder LaTeX-Dateien.

ModelNet und S3DIS sowie historische und aktuelle Versuchsprotokolle bleiben getrennt. Fehlende Bedingungen werden nicht ergänzt; ungültige Antworten zählen für Accuracy als falsch. Wiederholungen werden getrennt ausgewertet. Klassenwahrscheinlichkeiten werden unabhängig von den endgültigen Klassifikationsentscheidungen betrachtet. Gerundete Wahrscheinlichkeitsvektoren bleiben unverändert; historische unabhängige Ja/Nein-Scores werden nicht als Klassenverteilung interpretiert. Die gepaarten Wechsel beschreiben die ausgewählten Läufe und belegen allein keine Streuung über Wiederholungen.

Das versionierte `mmfm_tool/paper_figure_profile.json` ist Bestandteil des Programms und wird für die reproduzierbare Grafikerzeugung benötigt. Der Dateiname bleibt für bestehende Exporte kompatibel. Die optionale Pilot-Auswertung liest die historischen CSV-Dateien; die damalige Stable-Diffusion-Caption-Prüfung ist ausdrücklich als solche gekennzeichnet.

## Daten und lokale Einstellungen

Datensätze und Ergebnisse sind nicht im Quellcode enthalten. Die ignorierte `data_paths.local.json` kann lokale Pfade für `modelnet`, `s3dis`, `prepared`, `runs`, `legacy_results`, `historical_csv` und `paper_export` speichern. Umgebungsvariablen aus `.env.example` haben Vorrang. Datenquellen werden durch Auswertungen nicht verändert.

`.gitignore` lässt Programmdateien, benötigte Konfigurationen, Tests und Startdateien zu. Datensätze, Ergebnisse, erzeugte Grafiken, Sicherungen, lokale Pfade, virtuelle Umgebungen, Zugangsdaten und private Dokumente bleiben ausgeschlossen.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests für das optionale historische Archiv werden übersprungen, wenn es lokal nicht vorliegt. Die Tests verwenden keine kostenpflichtigen Modellanfragen.

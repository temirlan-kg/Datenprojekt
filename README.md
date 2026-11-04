# Vorhersage des Einkommens auf Basis von Bildungsmerkmalen (NEPScomp)

Datenprojekt im Studiengang Social Data Science & Communication, TH Nürnberg Georg Simon Ohm

| | |
|---|---|
| **Student** | Temirlan Anarkulov |
| **Betreuung** | Prof. Julia Plaß (TH Nürnberg), inhaltlich Michael Bergrab (LIfBi) |
| **Zeitraum** | Oktober 2026 bis Januar 2027 |


## Fragestellung

Lässt sich das spätere Einkommen einer Person aus ihren Bildungsmerkmalen vorhersagen?

1. Die Daten **deskriptiv analysieren** (Maßzahlen und Visualisierungen)
2. **Prädiktive Modelle** für das Einkommen entwickeln und verschiedene Methoden ausprobieren
3. Die Ansätze **evaluieren** und ein Modell auswählen

## Datengrundlage

- **Nationales Bildungspanel (NEPS)**, Aufbereitung **NEPScomp**, Startkohorte 6 (Erwachsene), Version 17.0.0
- Herausgeber: Leibniz-Institut für Bildungsverläufe (LIfBi), Bamberg, https://www.neps-data.de

| Datei | Verwendete Informationen |
|---|---|
| `SC6_Basics` | Geschlecht, Geburtsjahr, Bildung der Eltern |
| `SC6_Education` | Höchster Abschluss (ISCED-2011) |
| `SC6_spEmp` | Nettoeinkommen, Wochenarbeitszeit, Mini-Job |
| `SC6_spSchool` | Abschlussnote |

**Datenschutz:** Die NEPS-Daten unterliegen einem Datennutzungsvertrag und sind **nicht** in diesem Repository enthalten. `data/` und `output/` (Personendaten) sind per `.gitignore` ausgeschlossen. In `figures/` und `tables/` liegen ausschließlich aggregierte Ergebnisse.

## Projektstruktur

**Im Repository:**

```
education-income-prediction/
├── figures/                    # Abbildungen (nur aggregierte Ergebnisse)
├── tables/                     # Ergebnistabellen (nur aggregierte Ergebnisse)
├── .gitignore
├── 01_data_preparation.py      # Datensatz bauen: 1 Zeile pro Person
├── 02_descriptive_analysis.py  # Stichprobe, Verteilung, Eta², Spearman, Gruppenmittel
├── 03_models.py                # 5 Modelle, Train/Test-Split, Kreuzvalidierung
├── 04_evaluation.py            # Modellvergleich, Lasso-Effekte, Variable Importance
└── README.md
```

**Nur lokal (per `.gitignore` ausgeschlossen, Datenschutz):**

```
├── data/                       # NEPS-Rohdaten (.dta), Zugang nur über Datennutzungsvertrag
└── output/                     # Zwischenergebnisse mit Personendaten (.pkl, .csv, .joblib)
```

Wer den Code ausführen will, legt die NEPS-Dateien in `data/` ab. `output/` wird von den Skripten automatisch erstellt.

## Ausführung

Voraussetzungen: Python 3.12, Pakete:

```bash
pip install pandas numpy pyreadstat matplotlib seaborn scipy scikit-learn joblib
```

NEPS-Dateien (`.dta`) in den Ordner `data/` legen, dann die Skripte der Reihe nach ausführen:

```bash
python 01_data_preparation.py
python 02_descriptive_analysis.py
python 03_models.py
python 04_evaluation.py
```

## Vorgehen

| Schritt | Inhalt |
|---|---|
| Aufbereitung | NEPS-Missing-Codes (negative Werte) als fehlend behandelt; pro Person das letzte gültige Nettoeinkommen (ohne Mini-Jobs); höchster Abschluss in 7 Gruppen; Akademikerfamilie = mind. ein Elternteil mit FH-/Uni-Abschluss |
| Stichprobe | Nettoeinkommen 300–15.000 €, nur Vollzeit (≥ 35 Std./Woche): **n = 8.667** |
| Deskriptiv | Median/Quartile, Histogramm, Boxplots + Eta², Spearman-Korrelation, Gruppenmittelwerte |
| Modelle | Baseline (Median), Lineare Regression, Lasso, Random Forest, Gradient Boosting; Zielvariable log(Einkommen), Gütemaße in Euro |
| Evaluation | 80/20 Train/Test-Split, 5-fache Kreuzvalidierung, RMSE, MAE, R², Permutation Importance |

## Zentrale Ergebnisse

- Der höchste Abschluss erklärt allein ca. **18 %** der Einkommensunterschiede (η² = 0,18).
- Alle Modelle erreichen ein Test-R² von ca. **0,30** (RMSE ca. 1.320 €) und schlagen die Baseline deutlich.
- Komplexe Modelle (Random Forest, Gradient Boosting) sind kaum besser als die lineare Regression.
- Wichtigster Prädiktor ist der **Abschluss**; Abschlussnote und Bildung der Eltern tragen bei gegebenem Abschluss kaum zusätzlich bei.

## Einsatz von KI

Gemäß den Vorgaben ist gekennzeichnet, welche Teile mit KI erstellt wurden. KI-generierte Blöcke im Code sind so markiert:

```python
# >>> KI-generiert (Claude), von mir ausgeführt und nachvollzogen
...
# <<< Ende KI
```

| Datei | Zeilen | Art | Werkzeug |
|---|---|---|---|
| `README.md` | gesamte Datei | KI-generiert, von mir geprüft und angepasst | Claude (Anthropic) |
| `.gitignore` | gesamte Datei | selbst erstellt | – |
| `01_data_preparation.py` | gesamte Datei | KI-generiert, von mir ausgeführt und nachvollzogen | Claude (Anthropic) |
| `02_descriptive_analysis.py` | gesamte Datei | KI-generiert, von mir ausgeführt und nachvollzogen | Claude (Anthropic) |
| `03_models.py` | gesamte Datei | KI-generiert, von mir ausgeführt und nachvollzogen | Claude (Anthropic) |
| `04_evaluation.py` | gesamte Datei | KI-generiert, von mir ausgeführt und nachvollzogen | Claude (Anthropic) |

## Quellen

- Blossfeld, H.-P. & Roßbach, H.-G. (Hrsg.) (2019). *Education as a Lifelong Process: The German National Educational Panel Study (NEPS)*. 2. Aufl., Springer VS.
- NEPS-Netzwerk: Nationales Bildungspanel, Startkohorte 6 – Erwachsene. LIfBi, Bamberg. https://www.neps-data.de

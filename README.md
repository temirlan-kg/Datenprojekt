# Vorhersage des Einkommens auf Basis von Bildungsmerkmalen (NEPScomp)

Datenprojekt im Studiengang Social Data Science & Communication, TH Nürnberg Georg Simon Ohm
*Student:* Temirlan Anarkulov
*Betreuung:* Prof. Julia Plaß (TH Nürnberg), inhaltlich Michael Bergrab (LIfBi)
*Zeitraum:* Oktober 2026 bis Januar 2027
---

## Fragestellung

Lässt sich das spätere Einkommen einer Person aus ihren Bildungsmerkmalen vorhersagen?

Ziele des Projekts:

1. Die Daten **deskriptiv analysieren** (Maßzahlen und Visualisierungen)
2. **Prädiktive Modelle** für das Einkommen entwickeln und verschiedene Methoden ausprobieren
3. Die Ansätze **evaluieren** und ein Modell auswählen

## Datengrundlage

- **Nationales Bildungspanel (NEPS)**, Aufbereitung **NEPScomp** für die universitäre Lehre
- Startkohorte 6 (Erwachsene), aktuell genutzt: `SC6_spEmp_D_17-0-0.dta` (Erwerbsepisoden, 218.623 Zeilen × 146 Spalten)
- Herausgeber: Leibniz-Institut für Bildungsverläufe (LIfBi), Bamberg, https://www.neps-data.de

**Hinweis zum Datenschutz:** Die NEPS-Daten unterliegen einem Datennutzungsvertrag und sind **nicht** in diesem Repository enthalten. Der Ordner `data/` sowie alle `.dta`- und `.sav`-Dateien sind über `.gitignore` ausgeschlossen. Wer den Code ausführen will, braucht einen eigenen Datenzugang über das LIfBi.

## Projektstruktur

```
Datenprojekt/
├── README.md
├── .gitignore
├── data/        # NEPS-Rohdaten (lokal, nicht im Repo)
└── output/      # Zwischenergebnisse (lokal, nicht im Repo)
```

## Methoden (geplant)

| Phase | Methoden |
|---|---|
| Deskriptiv | Median, Quartile, Histogramm, Boxplots, Eta-Quadrat, Spearman-Korrelation, Gruppenmittelwerte |
| Modellierung | Lineare Regression, Lasso, Random Forest, Gradient Boosting |
| Evaluation | Train/Test-Split nach Personen-ID, Kreuzvalidierung, RMSE, R², Variable Importance |

## Ausführung

Voraussetzungen: Python 3 (Anaconda), `pandas`, `pyreadstat`

```bash
pip install pandas pyreadstat
```

Daten einlesen (Datei liegt lokal unter `data/`):

```python
import pyreadstat

emp, meta_emp = pyreadstat.read_dta("data/SC6_spEmp_D_17-0-0.dta", apply_value_formats=True)

# Spaltennamen mit Labels ergänzen, z. B. "ID_t | Target-ID"
neue_namen = {code: f"{code} | {text}" for code, text in meta_emp.column_names_to_labels.items()}
emp_lesbar = emp.rename(columns=neue_namen)
```

## Einsatz von KI

Gemäß den Vorgaben wird gekennzeichnet, welche Teile mit KI erstellt wurden. Im Code werden KI-generierte Blöcke so markiert:

```python
# >>> KI-generiert (Claude), angepasst
...
# <<< Ende KI
```

| Datei | Zeilen | Art | Werkzeug |
|---|---|---|---|
| `README.md` | gesamte Datei | KI-generiert, von mir geprüft und angepasst | Claude (Anthropic) |
| `.gitignore` | gesamte Datei | selbst erstellt | – |

*(Wird bei jeder neuen Datei ergänzt.)*

## Quellen

- NEPS-Netzwerk (Hrsg.): Nationales Bildungspanel, Startkohorte 6 – Erwachsene. Leibniz-Institut für Bildungsverläufe (LIfBi), Bamberg. https://www.neps-data.de

# ============================================================
# 02_descriptive_analysis.py – Deskriptive Analyse
#   1. Stichprobe festlegen (Ausreißer, Vollzeit)
#   2. Verteilung des Einkommens (Median, Quartile, Histogramm)
#   3. Einkommen nach Abschluss (Boxplot + Eta-Quadrat)
#   4. Abschlussnote und Einkommen (Spearman)
#   5. Einkommen nach Geschlecht und Akademikerfamilie (Gruppenmittel)
# >>> KI-generiert (Claude), von mir ausgeführt und nachvollzogen
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# --- Abschnitt 1: Pfade -------------------------------------
try:
    START = Path(__file__).resolve().parent
except NameError:
    START = Path.cwd()
ROOT = next((p for p in [START, *START.parents] if (p / "data").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError(f"Kein Ordner 'data' gefunden (gesucht ab {START})")
OUTPUT = ROOT / "output"          # lokal, Personendaten (in .gitignore)
FIGURES = ROOT / "figures"        # Abbildungen -> dürfen ins Repo (nur Zusammenfassungen)
TABELLEN = ROOT / "tables"      # Tabellen -> dürfen ins Repo (nur Zusammenfassungen)
FIGURES.mkdir(exist_ok=True)
TABELLEN.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# --- Abschnitt 2: Entscheidungen (hier zentral änderbar) ----
EINKOMMEN_MIN = 300      # darunter: kaum realistisch für einen Hauptjob
EINKOMMEN_MAX = 15000    # darüber: extreme Ausreißer
NUR_VOLLZEIT = True      # True = nur Personen mit >= 35 Wochenstunden


# --- Abschnitt 3: Daten laden & Stichprobe festlegen --------
df = pd.read_pickle(OUTPUT / "analyse_datensatz.pkl")
print("Start:", len(df))

df = df[df["einkommen"].between(EINKOMMEN_MIN, EINKOMMEN_MAX)]
print(f"nach Einkommensgrenzen {EINKOMMEN_MIN}–{EINKOMMEN_MAX} €:", len(df))

if NUR_VOLLZEIT:
    df = df[df["vollzeit"]]
    print("nur Vollzeit:", len(df))

df = df.dropna(subset=["abschluss"])
print("mit gültigem Abschluss:", len(df))

# Lesbare Labels für Grafiken
df["geschlecht_txt"] = df["geschlecht"].map({1: "männlich", 2: "weiblich"})
df["akademiker_txt"] = df["akademiker_eltern"].map({0: "nein", 1: "ja"})
df["log_einkommen"] = np.log(df["einkommen"])

# finale Stichprobe für die Modelle speichern
df.to_pickle(OUTPUT / "analyse_final.pkl")


# --- Abschnitt 4: Verteilung des Einkommens -----------------
verteilung = df["einkommen"].describe(percentiles=[0.25, 0.5, 0.75]).round(0)
print("\nVerteilung Nettoeinkommen:\n", verteilung)
print("Schiefe:", round(df["einkommen"].skew(), 2), "(> 0 = rechtsschief)")
verteilung.to_csv(TABELLEN / "01_verteilung_einkommen.csv", sep=";")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["einkommen"], bins=50, ax=ax[0])
ax[0].axvline(df["einkommen"].median(), color="red", linestyle="--", label="Median")
ax[0].set(title="Nettoeinkommen", xlabel="Euro pro Monat", ylabel="Anzahl")
ax[0].legend()
sns.histplot(df["log_einkommen"], bins=50, ax=ax[1])
ax[1].set(title="Nettoeinkommen (logarithmiert)", xlabel="log(Euro)", ylabel="Anzahl")
plt.tight_layout()
plt.savefig(FIGURES / "01_histogramm_einkommen.png")
plt.close()


# --- Abschnitt 5: Einkommen nach Abschluss + Eta-Quadrat -----
nach_abschluss = (df.groupby("abschluss")["einkommen"]
                    .agg(n="count", median="median", mittelwert="mean",
                         q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
                    .round(0))
print("\nEinkommen nach Abschluss:\n", nach_abschluss)
nach_abschluss.to_csv(TABELLEN / "02_einkommen_nach_abschluss.csv", sep=";")


def eta_quadrat(daten, gruppe, wert):
    """Anteil der Varianz von 'wert', der durch 'gruppe' erklärt wird."""
    gesamtmittel = daten[wert].mean()
    ss_gesamt = ((daten[wert] - gesamtmittel) ** 2).sum()
    gruppen = daten.groupby(gruppe)[wert]
    ss_zwischen = (gruppen.count() * (gruppen.mean() - gesamtmittel) ** 2).sum()
    return ss_zwischen / ss_gesamt


eta2 = eta_quadrat(df, "abschluss", "einkommen")
print(f"Eta² (Abschluss -> Einkommen): {eta2:.3f}")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="abschluss", y="einkommen", showfliers=False,
            order=sorted(df["abschluss"].unique()))   # Reihenfolge 1 bis 7
plt.xticks(rotation=30, ha="right")
plt.title(f"Nettoeinkommen nach höchstem Abschluss (η² = {eta2:.3f})")
plt.xlabel("")
plt.ylabel("Euro pro Monat")
plt.tight_layout()
plt.savefig(FIGURES / "02_boxplot_abschluss.png")
plt.close()


# --- Abschnitt 6: Abschlussnote & Einkommen (Spearman) ------
mit_note = df.dropna(subset=["abschlussnote"])
rho, p = spearmanr(mit_note["abschlussnote"], mit_note["einkommen"])
print(f"\nSpearman (Note -> Einkommen): rho = {rho:.3f}, p = {p:.4f}, n = {len(mit_note)}")
print("(Achtung: 1 = beste Note -> negatives rho heißt: bessere Note, mehr Einkommen)")

# Grafik mit Gruppenmedianen (keine Einzelpunkte -> Datenschutz)
mit_note = mit_note.assign(note_gerundet=mit_note["abschlussnote"].round(0).clip(1, 4))
note_tab = mit_note.groupby("note_gerundet")["einkommen"].agg(n="count", median="median").round(0)
print(note_tab)
note_tab.to_csv(TABELLEN / "03_einkommen_nach_note.csv", sep=";")

plt.figure(figsize=(7, 4))
sns.boxplot(data=mit_note, x="note_gerundet", y="einkommen", showfliers=False)
plt.title(f"Nettoeinkommen nach Abschlussnote (Spearman ρ = {rho:.3f})")
plt.xlabel("Abschlussnote (gerundet, 4 = 4 und schlechter)")
plt.ylabel("Euro pro Monat")
plt.tight_layout()
plt.savefig(FIGURES / "03_boxplot_note.png")
plt.close()


# --- Abschnitt 7: Gruppenmittel Geschlecht & Herkunft -------
for spalte, name in [("geschlecht_txt", "geschlecht"), ("akademiker_txt", "akademikerfamilie")]:
    tab = (df.groupby(spalte)["einkommen"]
             .agg(n="count", mittelwert="mean", median="median")
             .round(0))
    print(f"\nEinkommen nach {name}:\n", tab)
    tab.to_csv(TABELLEN / f"04_einkommen_nach_{name}.csv", sep=";")

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sns.barplot(data=df, x="geschlecht_txt", y="einkommen", estimator="mean", errorbar=("ci", 95), ax=ax[0])
ax[0].set(title="nach Geschlecht", xlabel="", ylabel="Mittleres Nettoeinkommen (€)")
sns.barplot(data=df.dropna(subset=["akademiker_txt"]), x="akademiker_txt", y="einkommen",
            estimator="mean", errorbar=("ci", 95), order=["nein", "ja"], ax=ax[1])
ax[1].set(title="Eltern Akademiker?", xlabel="", ylabel="")
plt.tight_layout()
plt.savefig(FIGURES / "04_gruppenmittel.png")
plt.close()

print("\nFertig. Abbildungen in figures/, Tabellen in tables/")
# <<< Ende KI
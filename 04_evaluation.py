# ============================================================
# 04_evaluation.py – Modelle vergleichen & interpretieren
#   1. Modellvergleich als Grafik (RMSE, R²)
#   2. Lasso: Welche Variablen bleiben übrig? Wie groß ist ihr Effekt (%)?
#   3. Variable Importance: Welche Variable ist am wichtigsten?
#   4. Vorhersage vs. Wirklichkeit für das beste Modell
# >>> KI-generiert (Claude), von mir ausgeführt und nachvollzogen
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance

# --- Abschnitt 1: Pfade & Laden -----------------------------
try:
    START = Path(__file__).resolve().parent
except NameError:
    START = Path.cwd()
ROOT = next((p for p in [START, *START.parents] if (p / "data").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError(f"Kein Ordner 'data' gefunden (gesucht ab {START})")
OUTPUT, FIGURES, TABELLEN = ROOT / "output", ROOT / "figures", ROOT / "tables"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

gespeichert = joblib.load(OUTPUT / "modelle.joblib")
modelle = gespeichert["modelle"]
X_test, y_test = gespeichert["X_test"], gespeichert["y_test"]
vergleich = pd.read_csv(TABELLEN / "05_modellvergleich.csv", sep=";")


# --- Abschnitt 2: Modellvergleich als Grafik -----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=vergleich, y="Modell", x="Test RMSE (€)", ax=ax[0], color="steelblue")
ax[0].set(title="Fehler auf Testdaten (kleiner = besser)", ylabel="")
sns.barplot(data=vergleich, y="Modell", x="Test R²", ax=ax[1], color="seagreen")
ax[1].set(title="Erklärte Varianz R² (größer = besser)", ylabel="", yticklabels=[])
plt.tight_layout()
plt.savefig(FIGURES / "05_modellvergleich.png")
plt.close()


# --- Abschnitt 3: Lasso-Koeffizienten ------------------------
# Das Modell lernt log(Einkommen). Ein Koeffizient b heißt dann ungefähr:
# "+1 Einheit -> Einkommen ändert sich um (exp(b) - 1) * 100 %".
# Bei den Abschluss-Spalten ist die Einheit "ja statt Haupt-/Realschule".
lasso_pipe = modelle["Lasso"].regressor_
namen = lasso_pipe.named_steps["vorverarbeitung"].get_feature_names_out()
koef = lasso_pipe.named_steps["modell"].coef_

# Zahlen-Spalten wurden skaliert -> Koeffizient gilt pro Standardabweichung.
# Zurückrechnen auf die ORIGINAL-Einheit (z. B. pro Notenstufe, pro Jahr, Frau statt Mann):
skalierer = (lasso_pipe.named_steps["vorverarbeitung"]
             .named_transformers_["num"].named_steps["skalieren"])
koef = koef.copy()
anzahl_kat = len(namen) - len(skalierer.scale_)
koef[anzahl_kat:] = koef[anzahl_kat:] / skalierer.scale_

lasso_tab = pd.DataFrame({"Variable": namen, "Koeffizient": koef})
lasso_tab["Variable"] = (lasso_tab["Variable"]
                         .str.replace("kat__abschluss_", "Abschluss: ")
                         .str.replace("num__missingindicator_", "fehlt: ")
                         .str.replace("num__", ""))
lasso_tab["Effekt (%)"] = ((np.exp(lasso_tab["Koeffizient"]) - 1) * 100).round(1)
lasso_tab["auf 0 gesetzt"] = lasso_tab["Koeffizient"].abs() < 1e-6
lasso_tab = lasso_tab.round(4)
print("Lasso-Koeffizienten:\n", lasso_tab.to_string(index=False))
print("alpha (gewählt):", round(lasso_pipe.named_steps["modell"].alpha_, 5))
lasso_tab.to_csv(TABELLEN / "06_lasso_koeffizienten.csv", sep=";", index=False)


# --- Abschnitt 4: Variable Importance (Permutation) ----------
# Idee: Man mischt EINE Spalte zufällig durch. Wird die Vorhersage dadurch viel
# schlechter, war die Spalte wichtig. Funktioniert für jedes Modell gleich.
wichtigkeit = []
for name in ["Lineare Regression", "Random Forest", "Gradient Boosting"]:
    pi = permutation_importance(modelle[name], X_test, y_test, n_repeats=10,
                                random_state=42, scoring="r2")
    wichtigkeit.append(pd.DataFrame({"Modell": name, "Variable": X_test.columns,
                                     "R²-Verlust": pi.importances_mean}))
wichtigkeit = pd.concat(wichtigkeit)
wichtigkeit["R²-Verlust"] = wichtigkeit["R²-Verlust"].round(4)
print("\nVariable Importance (R²-Verlust beim Durchmischen):")
print(wichtigkeit.pivot(index="Variable", columns="Modell", values="R²-Verlust")
                 .sort_values("Gradient Boosting", ascending=False))
wichtigkeit.to_csv(TABELLEN / "07_variable_importance.csv", sep=";", index=False)

plt.figure(figsize=(9, 4.5))
reihenfolge = (wichtigkeit.groupby("Variable")["R²-Verlust"].mean()
                          .sort_values(ascending=False).index)
sns.barplot(data=wichtigkeit, y="Variable", x="R²-Verlust", hue="Modell", order=reihenfolge)
plt.title("Variable Importance (Permutation, Testdaten)")
plt.xlabel("Verlust an R², wenn die Variable durchgemischt wird")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIGURES / "06_variable_importance.png")
plt.close()


# --- Abschnitt 5: Vorhersage vs. Wirklichkeit (bestes Modell) -
bestes = vergleich.loc[vergleich["Test RMSE (€)"].idxmin(), "Modell"]
print("\nBestes Modell (kleinster Test-RMSE):", bestes)
pred = modelle[bestes].predict(X_test)

# Keine Einzelpunkte (Datenschutz) -> Personen in 10 gleich große Gruppen nach
# Vorhersage einteilen und pro Gruppe Mittelwert von Vorhersage & Wirklichkeit zeigen
kal = pd.DataFrame({"vorhersage": pred, "wirklich": y_test.values})
kal["gruppe"] = pd.qcut(kal["vorhersage"], 10, labels=False)
kal = kal.groupby("gruppe")[["vorhersage", "wirklich"]].mean()

plt.figure(figsize=(5.5, 5))
plt.plot(kal["vorhersage"], kal["wirklich"], "o-", label=bestes)
grenze = [kal.min().min() * 0.9, kal.max().max() * 1.05]
plt.plot(grenze, grenze, "--", color="gray", label="perfekte Vorhersage")
plt.title("Vorhersage vs. Wirklichkeit (Testdaten, 10 Gruppen)")
plt.xlabel("Mittlere Vorhersage (€)")
plt.ylabel("Mittleres tatsächliches Einkommen (€)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "07_vorhersage_vs_wirklichkeit.png")
plt.close()

print("\nFertig. Neue Abbildungen 05–07 in figures/, Tabellen 06–07 in tables/")
# <<< Ende KI
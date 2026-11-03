# ============================================================
# 03_models.py – Prädiktive Modelle für das Nettoeinkommen
#   1. Daten laden, X (Prädiktoren) und y (Einkommen) festlegen
#   2. Train/Test-Split (80/20)
#   3. Vorverarbeitung (Kategorien -> 0/1, fehlende Werte, Skalierung)
#   4. 5 Modelle: Baseline, Lineare Regression, Lasso, Random Forest, Gradient Boosting
#   5. Kreuzvalidierung auf den Trainingsdaten
#   6. Test auf den 20 % unbekannten Daten
# >>> KI-generiert (Claude), von mir ausgeführt und nachvollzogen
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Abschnitt 1: Pfade -------------------------------------
try:
    START = Path(__file__).resolve().parent
except NameError:
    START = Path.cwd()
ROOT = next((p for p in [START, *START.parents] if (p / "data").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError(f"Kein Ordner 'data' gefunden (gesucht ab {START})")
OUTPUT = ROOT / "output"
TABELLEN = ROOT / "tables"
TABELLEN.mkdir(exist_ok=True)

ZUFALL = 42   # fester Zufallswert -> bei jedem Lauf dieselbe Aufteilung (reproduzierbar)


# --- Abschnitt 2: Daten laden, X und y festlegen ------------
df = pd.read_pickle(OUTPUT / "analyse_final.pkl")   # gefilterte Stichprobe aus 02

# Prädiktoren (X): Bildung + Kontrollvariablen
KATEGORISCH = ["abschluss"]                          # 7 Gruppen -> werden zu 0/1-Spalten
NUMERISCH = ["abschlussnote", "akademiker_eltern",   # Bildung / Herkunft
             "weiblich", "alter_einkommen",           # Kontrolle
             "jahr_einkommen"]                        # Kontrolle für Inflation / Zeit

df["weiblich"] = (df["geschlecht"] == 2).astype(int)  # 1 = Frau, 0 = Mann

X = df[KATEGORISCH + NUMERISCH]
y = df["einkommen"]                                   # in Euro

print("Personen:", len(df))
print("Fehlende Werte in X (%):")
print((X.isna().mean() * 100).round(1))


# --- Abschnitt 3: Train/Test-Split --------------------------
# 80 % zum Lernen, 20 % werden versteckt und erst ganz am Ende zum Prüfen benutzt.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=ZUFALL)
print(f"\nTraining: {len(X_train)}  |  Test: {len(X_test)}")


# --- Abschnitt 4: Vorverarbeitung ---------------------------
# a) Abschluss: Text-Kategorien -> 0/1-Spalten (One-Hot).
#    drop="first": erste Gruppe (Haupt-/Realschule) ist die Referenz.
# b) Zahlen: fehlende Werte mit dem Median füllen + eine Zusatzspalte
#    "Wert fehlte" (add_indicator). So verlieren wir die 52 % ohne Note nicht.
# c) Skalieren: alle Zahlen auf dieselbe Größenordnung (wichtig für Lasso).
vorverarbeitung = ColumnTransformer([
    ("kat", OneHotEncoder(drop="first", handle_unknown="ignore"), KATEGORISCH),
    ("num", Pipeline([
        ("fuellen", SimpleImputer(strategy="median", add_indicator=True)),
        ("skalieren", StandardScaler()),
    ]), NUMERISCH),
])


def mit_log(modell):
    """Modell lernt mit log(Einkommen), gibt Vorhersagen aber wieder in Euro aus."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("vorverarbeitung", vorverarbeitung), ("modell", modell)]),
        func=np.log, inverse_func=np.exp)


# --- Abschnitt 5: Die Modelle -------------------------------
modelle = {
    # Baseline: sagt für ALLE einfach den Median voraus -> jedes echte Modell muss besser sein
    "Baseline (Median)": mit_log(DummyRegressor(strategy="median")),
    "Lineare Regression": mit_log(LinearRegression()),
    # LassoCV sucht die Stärke der "Bestrafung" (alpha) selbst per Kreuzvalidierung
    "Lasso": mit_log(LassoCV(cv=5, random_state=ZUFALL)),
    "Random Forest": mit_log(RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, random_state=ZUFALL, n_jobs=-1)),
    "Gradient Boosting": mit_log(GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=ZUFALL)),
}


# --- Abschnitt 6: Kreuzvalidierung (nur Trainingsdaten) -----
# Trainingsdaten 5x unterschiedlich in Lern-/Prüfteil zerlegen -> stabileres Ergebnis
kfold = KFold(n_splits=5, shuffle=True, random_state=ZUFALL)
ergebnisse = []

for name, modell in modelle.items():
    cv = cross_validate(modell, X_train, y_train, cv=kfold,
                        scoring=["neg_root_mean_squared_error", "r2"])
    cv_rmse = -cv["test_neg_root_mean_squared_error"]
    cv_r2 = cv["test_r2"]

    # --- Abschnitt 7: Endgültig trainieren & auf Testdaten prüfen ---
    modell.fit(X_train, y_train)
    pred = modell.predict(X_test)

    ergebnisse.append({
        "Modell": name,
        "CV RMSE (€)": cv_rmse.mean(),
        "CV RMSE Std": cv_rmse.std(),
        "CV R²": cv_r2.mean(),
        "Test RMSE (€)": np.sqrt(mean_squared_error(y_test, pred)),
        "Test MAE (€)": mean_absolute_error(y_test, pred),
        "Test R²": r2_score(y_test, pred),
    })
    print(f"{name:20s} fertig")

ergebnisse = pd.DataFrame(ergebnisse).round(3)
pd.set_option("display.width", 200)
print("\nModellvergleich:\n", ergebnisse.to_string(index=False))
ergebnisse.to_csv(TABELLEN / "05_modellvergleich.csv", sep=";", index=False)


# --- Abschnitt 8: Speichern für 04_evaluation.py -------------
joblib.dump({"modelle": modelle, "X_test": X_test, "y_test": y_test,
             "X_train": X_train, "y_train": y_train},
            OUTPUT / "modelle.joblib")   # output/ -> bleibt lokal
print("\nGespeichert: output/modelle.joblib, tables/05_modellvergleich.csv")
# <<< Ende KI
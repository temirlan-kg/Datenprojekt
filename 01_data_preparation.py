# ============================================================
# 01_data_preparation.py – Datensatz für die Analyse bauen
# Ziel: ein Datensatz mit EINER Zeile pro Person (ID_t)
#   Teil 1: Basics + höchster Abschluss (Education)
#   Teil 2: Nettoeinkommen (spEmp)
# >>> KI-generiert (Claude), von mir ausgeführt und nachvollzogen
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

# --- Abschnitt 1: Pfade -------------------------------------
try:
    START = Path(__file__).resolve().parent         # Ordner, in dem dieses Skript liegt
except NameError:                                   # in der Konsole gibt es kein __file__
    START = Path.cwd()

# Projektordner = erster Ordner (von hier aus nach oben), der einen Ordner "data" enthält.
# So funktioniert es egal, ob das Skript im Hauptordner oder in src/ liegt.
ROOT = next((p for p in [START, *START.parents] if (p / "data").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError(f"Kein Ordner 'data' gefunden (gesucht ab {START})")
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


# --- Abschnitt 2: Hilfsfunktionen ---------------------------
def lade(name, spalten):
    """Lädt nur die gewünschten Spalten aus einer NEPS-Datei."""
    treffer = sorted(DATA.glob(f"*SC6_{name}_D_*.dta"))
    if not treffer:
        raise FileNotFoundError(f"Keine Datei für '{name}' in {DATA} gefunden")
    datei = treffer[0]
    try:
        df, meta = pyreadstat.read_dta(datei, usecols=spalten)
    except UnicodeDecodeError:                      # manche Dateien haben kaputte Umlaute
        df, meta = pyreadstat.read_dta(datei, usecols=spalten, encoding="latin1")
    return df, meta


def missing_zu_nan(df):
    """NEPS-Regel: alle negativen Werte sind fehlende Angaben (-97 verweigert, -98 weiß nicht, ...)."""
    zahlen = df.select_dtypes("number").columns.drop("ID_t", errors="ignore")
    df[zahlen] = df[zahlen].mask(df[zahlen] < 0)
    return df


# ============================================================
# TEIL 1: BASICS + HÖCHSTER ABSCHLUSS
# ============================================================

# --- Abschnitt 3: Basics laden (1 Zeile pro Person) ----------
basics_spalten = {
    "ID_t": "id",
    "t700001": "geschlecht",
    "t70000y": "geburtsjahr",
    "tx29000": "alter",
    "tx28102": "bildungsjahre",
    "t731301_g2": "bildung_mutter",   # CASMIN: Schule + Ausbildung + Hochschule
    "t731351_g2": "bildung_vater",
}
basics, meta_basics = lade("Basics", list(basics_spalten))

# ACHTUNG Ausnahme: -5 = "kein Schulabschluss" ist eine echte Antwort, kein Missing!
# Deshalb VOR missing_zu_nan auf 0 (= kein Abschluss) setzen.
for spalte in ["t731301_g2", "t731351_g2"]:
    basics[spalte] = basics[spalte].replace(-5, 0)

basics = missing_zu_nan(basics).rename(columns=basics_spalten)

# Akademikerfamilie: mind. ein Elternteil mit FH- (7) oder Uni-Abschluss (8)
mutter_aka = basics["bildung_mutter"].isin([7, 8])
vater_aka = basics["bildung_vater"].isin([7, 8])
beide_fehlen = basics["bildung_mutter"].isna() & basics["bildung_vater"].isna()
basics["akademiker_eltern"] = (mutter_aka | vater_aka).astype(float)
basics.loc[beide_fehlen, "akademiker_eltern"] = np.nan   # nichts bekannt -> fehlend

print("Basics:", basics.shape)
print("Personen doppelt?", basics["id"].duplicated().any())
print("Akademikerfamilie (1 = ja):")
print(basics["akademiker_eltern"].value_counts(dropna=False))


# --- Abschnitt 4: Höchster Abschluss aus Education -----------
edu, meta_edu = lade("Education", ["ID_t", "number", "tx28104"])
edu = missing_zu_nan(edu)

# Mehrere Zeilen pro Person -> nach Sortiernummer sortieren und die LETZTE nehmen
edu = (edu.dropna(subset=["tx28104"])
          .sort_values(["ID_t", "number"])
          .groupby("ID_t", as_index=False)
          .last())
edu = edu.rename(columns={"ID_t": "id", "tx28104": "isced2011"})[["id", "isced2011"]]


def isced_gruppe(code):
    """Fasst die ISCED-2011-Codes zu verständlichen Gruppen zusammen."""
    if pd.isna(code):
        return np.nan
    c = int(code)
    if c <= 254:
        return "1 Haupt-/Realschule oder weniger"   # 244 = Mittlere Reife
    if 340 <= c <= 344 or 440 <= c <= 444:
        return "2 Abitur ohne Ausbildung"
    if 350 <= c <= 354 or 450 <= c <= 454:
        return "3 Berufsausbildung"
    if 500 <= c <= 560 or c == 650:                 # 650 = Meister/Techniker
        return "4 Meister/Techniker/Fachschule"
    if c in (600, 640, 660):
        return "5 Bachelor"
    if 700 <= c <= 760:
        return "6 Master/Diplom/Staatsexamen"
    if 800 <= c <= 860:
        return "7 Promotion"
    return np.nan


edu["abschluss"] = edu["isced2011"].apply(isced_gruppe)

print("\nPersonen mit Abschluss:", len(edu))
print(edu["abschluss"].value_counts().sort_index())


# ============================================================
# TEIL 2: NETTOEINKOMMEN AUS spEmp
# ============================================================

# --- Abschnitt 5: Erwerbsepisoden laden ----------------------
emp_spalten = {
    "ID_t": "id",
    "wave": "welle",
    "spell": "spell",
    "subspell": "subspell",
    "ts23410_g1": "einkommen",      # Nettoeinkommen: offene Angabe, sonst Wert aus Spanne
    "ts23410": "einkommen_offen",   # nur offene Angabe (für späteren Robustheitscheck)
    "ts23223_g1": "stunden",        # tatsächliche Wochenarbeitszeit
    "ts23208": "minijob",           # 1 = ja, 2 = nein
}
emp, meta_emp = lade("spEmp", list(emp_spalten))
emp = emp.rename(columns=emp_spalten)

# welle ist eine Kategorie (Zwischenbefragung Corona = -20) -> nur echte Wellen behalten
emp = emp[emp["welle"] > 0]
emp = missing_zu_nan(emp)

print("\nspEmp Zeilen (Episoden x Wellen):", len(emp))


# --- Abschnitt 6: Filtern -----------------------------------
emp = emp.dropna(subset=["einkommen"])          # nur Zeilen mit gültigem Einkommen
emp = emp[emp["minijob"] != 1]                  # Mini-Jobs raus (fehlend = behalten)
emp = emp[emp["einkommen"] > 0]                 # 0 € ist kein Erwerbseinkommen

print("Zeilen mit gültigem Einkommen (ohne Mini-Job):", len(emp))
print("davon Personen:", emp["id"].nunique())


# --- Abschnitt 7: Ein Einkommen pro Person ------------------
# Regel: das LETZTE gültige Einkommen (neueste Welle).
# Hat jemand in derselben Welle mehrere Jobs -> den mit den meisten Stunden (Hauptjob).
emp = emp.sort_values(["id", "welle", "stunden"], na_position="first")
eink = emp.drop_duplicates(subset="id", keep="last")   # ganze letzte ZEILE pro Person
# Achtung: groupby().last() wäre hier falsch – es nimmt pro SPALTE den letzten
# nicht-fehlenden Wert und mischt so Werte aus verschiedenen Zeilen.

# Jahr der Welle: Welle 1 = 2007, Welle 2 = 2009/10, Welle 3 = 2010/11, ...
eink["jahr_einkommen"] = np.where(eink["welle"] == 1, 2007, 2007 + eink["welle"])

eink = eink[["id", "einkommen", "einkommen_offen", "stunden", "welle", "jahr_einkommen"]]

print("\nEinkommen pro Person:", eink.shape)
print(eink["einkommen"].describe().round(0))


# ============================================================
# TEIL 3: ABSCHLUSSNOTE AUS spSchool
# ============================================================

# --- Abschnitt 8: Note des letzten Schulabschlusses ----------
schule_spalten = {
    "ID_t": "id",
    "subspell": "subspell",
    "ts1112y_g1": "endjahr",        # Endjahr der Schulepisode
    "ts11209": "schulabschluss",    # Art des Schulabschlusses
    "ts11218": "abschlussnote",     # Gesamtnote im Abschlusszeugnis (1,0 bis 6,0)
}
schule, meta_schule = lade("spSchool", list(schule_spalten))
schule = schule.rename(columns=schule_spalten)
schule = schule[schule["subspell"] == 0]      # 0 = ganze Episode (Teilepisoden sind doppelt)
schule = missing_zu_nan(schule)

# nur Noten im gültigen Notenbereich
schule = schule[schule["abschlussnote"].between(0.7, 6.0)]

# pro Person: Note der LETZTEN Schulepisode (= höchster Schulabschluss)
schule = schule.sort_values(["id", "endjahr"], na_position="first")
note = schule.drop_duplicates(subset="id", keep="last")[["id", "abschlussnote"]]

print("\nPersonen mit Abschlussnote:", len(note))
print(note["abschlussnote"].describe().round(2))


# ============================================================
# ZUSAMMENFÜHREN
# ============================================================

# --- Abschnitt 9: Alles über die Personen-ID verbinden -------
df = (basics
      .merge(edu, on="id", how="left")
      .merge(note, on="id", how="left")     # left = Note darf fehlen
      .merge(eink, on="id", how="inner"))   # inner = nur Personen MIT Einkommen

# Alter zum Zeitpunkt der Einkommensangabe (nicht zum letzten Interview)
df["alter_einkommen"] = df["jahr_einkommen"] - df["geburtsjahr"]

# Vollzeit-Markierung (Entscheidung, ob nur Vollzeit, kommt in der Analyse)
VOLLZEIT_AB = 35
df["vollzeit"] = df["stunden"] >= VOLLZEIT_AB

print("\nAnalysedatensatz:", df.shape)
print("davon Vollzeit (>= 35 Std.):", df["vollzeit"].sum())
print("\nFehlende Werte pro Spalte (%):")
print((df.isna().mean() * 100).round(1))

# Speichern (output/ ist in .gitignore -> bleibt lokal)
df.to_pickle(OUTPUT / "analyse_datensatz.pkl")
print("\nGespeichert: output/analyse_datensatz.pkl")

# Zusätzlich als CSV zum Anschauen in Excel (bleibt ebenfalls lokal)
df.to_csv(OUTPUT / "analyse_datensatz.csv", index=False, sep=";", encoding="utf-8-sig")
# <<< Ende KI
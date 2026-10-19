"""Wczytanie danych zrodlowych: wyniki olimpijskie i wskazniki Banku Swiatowego.

Modul sprowadza oba zrodla do wspolnego formatu, nie przeksztalcajac ich
merytorycznie - transformacje sa w `pre_processing`.
"""

from pathlib import Path

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Korzen projektu wyznaczony przez szukanie w gore pliku src/data_load.py.

    Rozpoznawanie po nazwie katalogu roboczego zawodzi, bo edytory oparte na
    VS Code ustawiaja katalog roboczy notebooka roznie.
    """
    root = (start or Path.cwd()).resolve()
    while not (root / "src" / "data_load.py").is_file():
        if root == root.parent:
            raise RuntimeError(
                f"Nie znaleziono korzenia projektu startując z {start or Path.cwd()}."
            )
        root = root.parent
    return root


PROJECT_ROOT = find_project_root()
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Zakres: tylko letnie igrzyska. Panel co 4 lata jest spojny, a WDI to dane roczne.
EDITION_TYPE = "Summer"

# Seria WDI -> nazwa kolumny w panelu.
SERIES = {"NY.GDP.PCAP.KD": "gdp_pc", "SP.POP.TOTL": "pop"}

WDI_FILE = RAW / "wdi_worldbank.csv"

# Kody NOC (Olympedia) roznia sie od ISO3 (Bank Swiatowy) dla czesci krajow.
NOC2ISO = {
    "ALG": "DZA", "ANG": "AGO", "ANT": "ATG", "ARU": "ABW", "ASA": "ASM",
    "BAH": "BHS", "BAN": "BGD", "BAR": "BRB", "BIZ": "BLZ", "BOT": "BWA",
    "BRU": "BRN", "BUL": "BGR", "CAM": "KHM", "CAY": "CYM", "CGO": "COG",
    "CHA": "TCD", "CHI": "CHL", "CRC": "CRI", "CRO": "HRV", "DEN": "DNK",
    "ESA": "SLV", "FIJ": "FJI", "GAM": "GMB", "GBS": "GNB", "GEQ": "GNQ",
    "GER": "DEU", "GRE": "GRC", "GRN": "GRD", "GUA": "GTM", "GUI": "GIN",
    "HAI": "HTI", "HON": "HND", "INA": "IDN", "IRI": "IRN", "ISV": "VIR",
    "KSA": "SAU", "KUW": "KWT", "LAT": "LVA", "LBA": "LBY", "LES": "LSO",
    "LIB": "LBN", "MAD": "MDG", "MAS": "MYS", "MAW": "MWI", "MGL": "MNG",
    "MRI": "MUS", "MTN": "MRT", "MYA": "MMR", "NCA": "NIC", "NED": "NLD",
    "NEP": "NPL", "NGR": "NGA", "NIG": "NER", "OMA": "OMN", "PAR": "PRY",
    "PHI": "PHL", "PLE": "PSE", "POR": "PRT", "PUR": "PRI", "RSA": "ZAF",
    "SAM": "WSM", "SEY": "SYC", "SIN": "SGP", "SLO": "SVN", "SOL": "SLB",
    "SRI": "LKA", "SUD": "SDN", "SUI": "CHE", "TAN": "TZA", "TOG": "TGO",
    "UAE": "ARE", "URU": "URY", "VAN": "VUT", "VIE": "VNM", "VIN": "VCT",
    "ZAM": "ZMB", "ZIM": "ZWE",
}


def load_event_results() -> pd.DataFrame:
    """Wyniki na poziomie zawodnik x konkurencja (`medal`, `isTeamSport`)."""
    return pd.read_csv(RAW / "Olympic_Athlete_Event_Results.csv", encoding="utf-8-sig")


def load_summer_games() -> pd.DataFrame:
    """Meta letnich igrzysk: `edition_id`, `year`, kraj gospodarza (`host_noc`)."""
    games = pd.read_csv(RAW / "Olympics_Games.csv", encoding="utf-8-sig")

    def edition_type(s):
        s = str(s)
        if "Summer" in s:
            return "Summer"
        if "Winter" in s:
            return "Winter"
        return "Other"

    games["etype"] = games["edition"].map(edition_type)
    games["year"] = pd.to_numeric(games["year"], errors="coerce")
    host = games[["edition_id", "year", "etype", "country_noc"]].rename(
        columns={"country_noc": "host_noc"}
    )

    summer = host[host["etype"] == EDITION_TYPE].dropna(subset=["year"]).copy()
    summer["year"] = summer["year"].astype(int)
    return summer


def map_noc_to_iso3(noc: pd.Series) -> pd.Series:
    """NOC -> ISO3. Kody nieobecne w slowniku przechodza bez zmiany."""
    return noc.map(lambda x: NOC2ISO.get(x, x))


def fetch_wdi(iso_list: list[str], year_min: int, year_max: int) -> pd.DataFrame:
    """Pobranie PKB per capita i populacji z API Banku Swiatowego.

    Wymaga pakietu `wbgapi`. Bez dostepu do sieci albo bez tego pakietu
    nalezy uzyc `load_wdi`, ktora czyta zapisana kopie.
    """
    import wbgapi as wb

    d = wb.data.DataFrame(
        list(SERIES),
        economy=iso_list,
        time=range(max(year_min, 1960), year_max + 1),
        numericTimeKeys=True,
        columns="series",
        skipBlanks=False,
    )
    d = d.reset_index().rename(columns={"economy": "iso3", "time": "year"})
    return d.rename(columns=SERIES)


def load_wdi(
    iso_list: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """WDI z zapisanego pliku; przy `refresh=True` proba pobrania z API.

    Pobranie wymaga `wbgapi` (patrz requirements.txt) i dostepu do sieci.
    Nieudane pobranie jest zglaszane wprost, a nie maskowane odczytem pliku -
    inaczej nie wiadomo, czy dane sa swieze.
    """
    if refresh:
        if iso_list is None or year_min is None or year_max is None:
            raise ValueError("Odświeżenie wymaga iso_list, year_min i year_max.")
        wdi = fetch_wdi(iso_list, year_min, year_max)
        wdi.to_csv(WDI_FILE, index=False, encoding="utf-8")
        print(f"WDI pobrane z API: {wdi.shape} -> zapisano do {WDI_FILE.name}")
        return wdi

    if not WDI_FILE.is_file():
        raise FileNotFoundError(
            f"Brak {WDI_FILE}. Uruchom load_wdi(..., refresh=True), "
            "żeby pobrać dane z Banku Światowego."
        )
    return pd.read_csv(WDI_FILE)

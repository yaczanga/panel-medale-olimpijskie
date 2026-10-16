"""Budowa panelu kraj x igrzyska z surowych wyników olimpijskich i WDI."""

import numpy as np
import pandas as pd

from src.data_load import PROCESSED, map_noc_to_iso3

MEDALS = ["Gold", "Silver", "Bronze"]

# Zmienne wchodzące do równania; braki w którejkolwiek usuwają obserwację.
MODEL_VARS = ["medals", "log_pop", "log_gdp_pc", "host", "medals_lag"]

PANEL_FILE = PROCESSED / "panel_analityczny.csv"


def count_medals(event_results: pd.DataFrame) -> pd.DataFrame:
    """Liczba medali kraju na igrzyskach, z deduplikacją medali drużynowych.

    Jeden medal = unikalna krotka (edition_id, country_noc, event, medal).
    Drużyna jedenastu piłkarzy to jedno złoto, nie jedenaście.
    """
    rows = event_results[event_results["medal"].isin(MEDALS)]
    unique = rows.drop_duplicates(subset=["edition_id", "country_noc", "event", "medal"])
    return (
        unique.groupby(["edition_id", "country_noc"])
        .size()
        .rename("medals")
        .reset_index()
    )


def build_panel(event_results: pd.DataFrame, summer: pd.DataFrame) -> pd.DataFrame:
    """Panel kraj x letnie igrzyska: medale, rok, gospodarz, opóźnienie.

    Panel powstaje z listy **uczestnictw**, nie z listy medali. Kraj, który
    wystartował i nie zdobył medalu, ma `medals = 0` - model liczebności
    wymaga tych zer, a lista medali by je pominęła.
    """
    ev = event_results.merge(summer[["edition_id"]], on="edition_id", how="inner")

    medals = count_medals(ev)
    participation = ev[["edition_id", "country_noc"]].drop_duplicates().reset_index(drop=True)

    panel = participation.merge(medals, on=["edition_id", "country_noc"], how="left")
    panel["medals"] = panel["medals"].fillna(0).astype(int)

    panel = panel.merge(summer[["edition_id", "year", "host_noc"]], on="edition_id", how="left")
    panel["host"] = (panel["country_noc"] == panel["host_noc"]).astype(int)

    return add_lag(panel)


def add_lag(panel: pd.DataFrame) -> pd.DataFrame:
    """`medals_lag`: medale kraju na jego poprzednich letnich igrzyskach.

    Przesunięcie jest liczone **wewnątrz kraju** po uporządkowaniu po roku,
    więc pierwsza obserwacja każdego kraju ma brak, a nie wartość sąsiada.
    """
    panel = panel.sort_values(["country_noc", "year"]).reset_index(drop=True)
    panel["medals_lag"] = panel.groupby("country_noc")["medals"].shift(1)
    return panel


def merge_wdi(panel: pd.DataFrame, wdi: pd.DataFrame) -> pd.DataFrame:
    """Dołączenie PKB per capita i populacji po ISO3 i roku; logarytmy."""
    panel = panel.copy()
    panel["iso3"] = map_noc_to_iso3(panel["country_noc"])

    df = panel.merge(wdi, on=["iso3", "year"], how="left")
    df["log_pop"] = np.log(df["pop"])
    df["log_gdp_pc"] = np.log(df["gdp_pc"])
    return df


def to_model_frame(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Usunięcie obserwacji z brakami w zmiennych modelu; opcjonalny zapis."""
    out = df.dropna(subset=MODEL_VARS).reset_index(drop=True).copy()
    out["country"] = out["country_noc"].astype("category")

    if save:
        PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(PANEL_FILE, index=False, encoding="utf-8")
        print(f"Zapisano panel analityczny: {PANEL_FILE.name} ({len(out)} obserwacji)")
    return out


def load_model_frame() -> pd.DataFrame:
    """Gotowy panel analityczny z data/processed - bez przeliczania od surowych."""
    if not PANEL_FILE.is_file():
        raise FileNotFoundError(
            f"Brak {PANEL_FILE}. Uruchom notebooks/01_panel_medale.ipynb, "
            "żeby zbudować panel z danych surowych."
        )
    df = pd.read_csv(PANEL_FILE)
    df["country"] = df["country_noc"].astype("category")
    return df


def add_mundlak_means(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Średnie wewnątrzkrajowe regresorów (`*_bar`) - urządzenie Mundlaka."""
    means = df.groupby("country_noc")[variables].transform("mean")
    means.columns = [c + "_bar" for c in variables]
    return pd.concat([df, means], axis=1)


def describe_panel(df: pd.DataFrame) -> dict:
    """Liczby opisujące panel - te same, które trafiają do README i tabel."""
    return {
        "N": len(df),
        "krajow": df["country_noc"].nunique(),
        "edycji": df["edition_id"].nunique(),
        "rok_min": int(df["year"].min()),
        "rok_max": int(df["year"].max()),
        "udzial_zer": float((df["medals"] == 0).mean()),
        "srednia": float(df["medals"].mean()),
        "wariancja": float(df["medals"].var()),
    }

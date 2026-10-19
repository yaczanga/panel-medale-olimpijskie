"""Testy gwarancji metodologicznych budowy panelu.

Kazdy test dziala na malej probie syntetycznej, bez dotykania danych zrodlowych.
Dzieki temu wykonuja sie w sekundy i dzialaja w CI, gdzie 95 MB wynikow
olimpijskich nie istnieje.

Sprawdzane sa te wlasnosci, na ktorych stoja wnioski: deduplikacja medali
druzynowych, obecnosc zer dla uczestnikow bez medalu, brak przecieku miedzy
krajami w opoznieniu oraz jednoznacznosc mapowania NOC na ISO3.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_load import NOC2ISO, map_noc_to_iso3
from src.pre_processing import (
    add_lag,
    add_mundlak_means,
    build_panel,
    count_medals,
    merge_wdi,
    to_model_frame,
)


def make_summer(editions):
    """Meta letnich igrzysk: (edition_id, rok, gospodarz)."""
    return pd.DataFrame(
        [{"edition_id": e, "year": y, "etype": "Summer", "host_noc": h}
         for e, y, h in editions]
    )


def make_events(rows):
    """Wyniki zawodnik x konkurencja: (edition_id, kraj, konkurencja, medal)."""
    return pd.DataFrame(
        [{"edition_id": e, "country_noc": c, "event": ev, "medal": m}
         for e, c, ev, m in rows]
    )


class TestDeduplikacjaMedaliDruzynowych:
    """Jeden medal druzynowy to jeden medal, nie tyle, ilu bylo zawodnikow."""

    def test_druzyna_jedenastu_pilkarzy_daje_jedno_zloto(self):
        events = make_events(
            [(1, "POL", "Football", "Gold")] * 11
        )
        out = count_medals(events)
        assert len(out) == 1
        assert out.loc[0, "medals"] == 1

    def test_rozne_kolory_w_tej_samej_konkurencji_licza_sie_osobno(self):
        events = make_events(
            [(1, "POL", "Rowing", "Gold")] * 4
            + [(1, "POL", "Rowing", "Silver")] * 4
        )
        assert count_medals(events).loc[0, "medals"] == 2

    def test_ta_sama_konkurencja_na_roznych_igrzyskach_liczy_sie_osobno(self):
        events = make_events(
            [(1, "POL", "Football", "Gold")] * 11
            + [(2, "POL", "Football", "Gold")] * 11
        )
        out = count_medals(events).set_index("edition_id")
        assert out.loc[1, "medals"] == 1
        assert out.loc[2, "medals"] == 1

    def test_medale_indywidualne_nie_sa_scalane(self):
        """Dwoch roznych plywakow w dwoch konkurencjach to dwa medale."""
        events = make_events(
            [(1, "POL", "Swimming 100m", "Gold"), (1, "POL", "Swimming 200m", "Gold")]
        )
        assert count_medals(events).loc[0, "medals"] == 2


class TestZeraDlaUczestnikowBezMedalu:
    """Panel musi zawierac zera, bo model liczebnosci na nich stoi."""

    def test_kraj_bez_medalu_ma_zero_a_nie_brak_wiersza(self):
        events = make_events(
            [(1, "USA", "Swimming", "Gold"), (1, "POL", "Swimming", np.nan)]
        )
        panel = build_panel(events, make_summer([(1, 2000, "AUS")]))
        assert set(panel["country_noc"]) == {"USA", "POL"}
        assert panel.set_index("country_noc").loc["POL", "medals"] == 0

    def test_udzial_zer_liczony_z_uczestnictw_nie_z_medali(self):
        """Trzy kraje startuja, jeden zdobywa medal - udzial zer to dwie trzecie."""
        events = make_events(
            [(1, "USA", "Swimming", "Gold")]
            + [(1, c, "Swimming", np.nan) for c in ("POL", "KEN")]
        )
        panel = build_panel(events, make_summer([(1, 2000, "AUS")]))
        assert (panel["medals"] == 0).mean() == pytest.approx(2 / 3)

    def test_medal_bez_uczestnictwa_nie_tworzy_wiersza_widma(self):
        """Panel wychodzi z uczestnictw, wiec liczba wierszy to liczba par."""
        events = make_events(
            [(1, "USA", "Swimming", "Gold"), (1, "USA", "Rowing", "Silver")]
        )
        panel = build_panel(events, make_summer([(1, 2000, "AUS")]))
        assert len(panel) == 1
        assert panel.loc[0, "medals"] == 2


class TestGospodarz:
    def test_gospodarz_rozpoznany_po_kodzie_kraju(self):
        events = make_events(
            [(1, "AUS", "Swimming", "Gold"), (1, "POL", "Swimming", np.nan)]
        )
        panel = build_panel(events, make_summer([(1, 2000, "AUS")])).set_index("country_noc")
        assert panel.loc["AUS", "host"] == 1
        assert panel.loc["POL", "host"] == 0


class TestOpoznienieBezPrzecieku:
    """`medals_lag` nie moze przeciekac miedzy krajami ani z przyszlosci."""

    def test_pierwsza_obserwacja_kraju_ma_brak(self):
        panel = pd.DataFrame(
            {"country_noc": ["POL", "POL"], "year": [2000, 2004], "medals": [3, 5]}
        )
        out = add_lag(panel).set_index("year")
        assert pd.isna(out.loc[2000, "medals_lag"])
        assert out.loc[2004, "medals_lag"] == 3

    def test_opoznienie_nie_przecieka_miedzy_krajami(self):
        """Pierwszy wiersz USA nie moze dostac ostatniej wartosci POL."""
        panel = pd.DataFrame(
            {
                "country_noc": ["POL", "POL", "USA", "USA"],
                "year": [2000, 2004, 2000, 2004],
                "medals": [3, 5, 100, 110],
            }
        )
        out = add_lag(panel).set_index(["country_noc", "year"])
        assert pd.isna(out.loc[("USA", 2000), "medals_lag"])
        assert out.loc[("USA", 2004), "medals_lag"] == 100
        assert out.loc[("POL", 2004), "medals_lag"] == 3

    def test_kolejnosc_wejsciowa_nie_wplywa_na_opoznienie(self):
        """Przesuniecie musi isc po roku, nie po kolejnosci wierszy w pliku."""
        rows = {
            "country_noc": ["POL", "POL", "POL"],
            "year": [2008, 2000, 2004],
            "medals": [7, 3, 5],
        }
        out = add_lag(pd.DataFrame(rows)).set_index("year")
        assert pd.isna(out.loc[2000, "medals_lag"])
        assert out.loc[2004, "medals_lag"] == 3
        assert out.loc[2008, "medals_lag"] == 5

    def test_opoznienie_nie_pochodzi_z_przyszlosci(self):
        panel = pd.DataFrame(
            {"country_noc": ["POL"] * 3, "year": [2000, 2004, 2008], "medals": [3, 5, 7]}
        )
        out = add_lag(panel)
        assert list(out["medals_lag"].dropna()) == [3, 5]


class TestMapowanieNocIso:
    def test_kod_nieobecny_w_slowniku_przechodzi_bez_zmiany(self):
        assert list(map_noc_to_iso3(pd.Series(["POL", "USA"]))) == ["POL", "USA"]

    def test_znane_rozbieznosci_sa_tlumaczone(self):
        got = map_noc_to_iso3(pd.Series(["GER", "NED", "SUI", "RSA"]))
        assert list(got) == ["DEU", "NLD", "CHE", "ZAF"]

    def test_mapowanie_jest_jednoznaczne(self):
        """Dwa kody NOC nie moga wskazywac tego samego ISO3 - to by scalilo kraje."""
        values = list(NOC2ISO.values())
        assert len(values) == len(set(values))

    def test_mapowanie_nie_jest_tozsamosciowe(self):
        """Wpis NOC -> ten sam NOC bylby zbedny i mylacy."""
        assert not [k for k, v in NOC2ISO.items() if k == v]


class TestProbaModelowa:
    def test_braki_w_zmiennych_modelu_usuwaja_obserwacje(self):
        df = pd.DataFrame(
            {
                "country_noc": ["POL", "USA", "KEN"],
                "edition_id": [1, 1, 1],
                "year": [2000] * 3,
                "medals": [3, 5, 1],
                "medals_lag": [1.0, np.nan, 0.0],
                "log_pop": [17.0, 19.0, 17.5],
                "log_gdp_pc": [9.0, 10.5, np.nan],
                "host": [0, 0, 0],
            }
        )
        out = to_model_frame(df)
        assert list(out["country_noc"]) == ["POL"]

    def test_srednie_mundlaka_liczone_w_obrebie_kraju(self):
        df = pd.DataFrame(
            {
                "country_noc": ["POL", "POL", "USA", "USA"],
                "log_gdp_pc": [9.0, 11.0, 10.0, 20.0],
            }
        )
        out = add_mundlak_means(df, ["log_gdp_pc"])
        assert list(out["log_gdp_pc_bar"]) == [10.0, 10.0, 15.0, 15.0]

    def test_merge_wdi_po_iso3_a_nie_po_noc(self):
        """Kraj o kodzie NOC roznym od ISO3 musi dostac swoje dane WDI."""
        panel = pd.DataFrame({"country_noc": ["GER"], "year": [2000]})
        wdi = pd.DataFrame({"iso3": ["DEU"], "year": [2000], "gdp_pc": [30000.0], "pop": [82e6]})
        out = merge_wdi(panel, wdi)
        assert out.loc[0, "gdp_pc"] == 30000.0
        assert out.loc[0, "log_gdp_pc"] == pytest.approx(np.log(30000.0))

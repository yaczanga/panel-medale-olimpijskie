"""Testy wnioskowania na syntetycznym panelu o znanym procesie generującym.

Testy nie sprawdzają konkretnych oszacowań - to byłoby testowanie statsmodels.
Sprawdzają własności, które muszą zachodzić: że diagnostyka dyspersji reaguje
na dyspersję, że test Mundlaka wykrywa skorelowany efekt kraju, i że tabele
wynikowe nie zawierają liczb innych niż w samych oszacowaniach.

Generator panelu jest w conftest.py.
"""

import numpy as np
import pytest

from src.inference import compare_gdp_effect, dispersion_parts, irr_table, mundlak_test
from src.model import TERMS, cre_poisson, fit_all, pooled_poisson

# Powyżej tego progu uznajemy dyspersję za nadmierną (jak w inference).
PROG_DYSPERSJI = 1.5


@pytest.fixture(scope="module")
def fits(panel_czysty):
    return fit_all(panel_czysty)


class TestDyspersja:
    def test_skladniki_sa_zgodne_z_ilorazem(self, panel_czysty):
        chi2, df, ratio = dispersion_parts(pooled_poisson(panel_czysty))
        assert ratio == pytest.approx(chi2 / df)
        assert df > 0

    def test_poprawnie_wyspecyfikowany_poisson_daje_dyspersje_okolo_jedynki(
        self, panel_czysty
    ):
        """Bez efektu kraju i bez domieszki Var(Y) = E(Y), więc iloraz ~ 1."""
        _, _, ratio = dispersion_parts(pooled_poisson(panel_czysty))
        assert 0.7 < ratio < PROG_DYSPERSJI

    def test_domieszka_gamma_podnosi_dyspersje_powyzej_progu(
        self, panel_czysty, panel_nadmierna_dyspersja
    ):
        _, _, czysty = dispersion_parts(pooled_poisson(panel_czysty))
        _, _, nb = dispersion_parts(pooled_poisson(panel_nadmierna_dyspersja))
        assert nb > czysty
        assert nb > PROG_DYSPERSJI

    def test_nieuwzgledniony_efekt_kraju_tez_podnosi_dyspersje(
        self, panel_czysty, panel_skorelowany_efekt
    ):
        """Heterogeniczność pominięta w równaniu ujawnia się jako dyspersja."""
        _, _, czysty = dispersion_parts(pooled_poisson(panel_czysty))
        _, _, z_efektem = dispersion_parts(pooled_poisson(panel_skorelowany_efekt))
        assert z_efektem > czysty


class TestTabelaIrr:
    def test_irr_jest_eksponentem_wspolczynnika(self, fits):
        res = fits["pooled_cluster"]
        tab = irr_table(res.params, res.bse)
        assert np.allclose(tab["IRR"], np.exp(tab["coef"]))

    def test_statystyka_z_to_iloraz_wspolczynnika_i_bledu(self, fits):
        res = fits["pooled_cluster"]
        tab = irr_table(res.params, res.bse)
        assert np.allclose(tab["z"], tab["coef"] / tab["se"])

    def test_wartosc_p_jest_w_przedziale_jednostkowym(self, fits):
        res = fits["pooled_cluster"]
        assert irr_table(res.params, res.bse)["p"].between(0, 1).all()

    def test_tabela_zawiera_dokladnie_zmienne_modelu(self, fits):
        res = fits["pooled_cluster"]
        assert list(irr_table(res.params, res.bse).index) == TERMS


class TestPorownanieEstymatorow:
    def test_tabela_ma_wiersz_na_kazdy_estymator(self, fits):
        comp = compare_gdp_effect(fits)
        assert len(comp) == 5
        assert comp["model"].is_unique

    def test_wspolczynniki_zgadzaja_sie_ze_zrodlowymi_modelami(self, fits):
        """Tabela nie może zawierać liczb innych niż w samych oszacowaniach."""
        comp = compare_gdp_effect(fits).set_index("model")
        assert comp.loc["Pooled Poisson (klaster)", "coef_log_gdp_pc"] == pytest.approx(
            fits["pooled_cluster"].params["log_gdp_pc"]
        )
        assert comp.loc["FE-Poisson", "coef_log_gdp_pc"] == pytest.approx(
            fits["fe"].coef()["log_gdp_pc"]
        )
        assert comp.loc["CRE / Mundlak", "coef_log_gdp_pc"] == pytest.approx(
            fits["cre"].params["log_gdp_pc"]
        )
        assert comp.loc["PA Poisson (GEE)", "coef_log_gdp_pc"] == pytest.approx(
            fits["pa"].params["log_gdp_pc"]
        )

    def test_bledy_standardowe_tez_pochodza_z_modeli(self, fits):
        comp = compare_gdp_effect(fits).set_index("model")
        assert comp.loc["FE-Poisson", "se"] == pytest.approx(
            fits["fe"].se()["log_gdp_pc"]
        )

    def test_irr_dla_wzrostu_pkb_o_dziesiec_procent(self, fits):
        comp = compare_gdp_effect(fits)
        assert np.allclose(
            comp["IRR_+10%PKB"], np.exp(comp["coef_log_gdp_pc"] * np.log(1.10))
        )

    def test_brak_brakow_w_tabeli_porownawczej(self, fits):
        """Brak oznaczałby, że jeden z estymatorów nie zwrócił współczynnika."""
        assert compare_gdp_effect(fits).notna().all().all()


class TestTestMundlaka:
    def test_zwraca_statystyke_i_wartosc_p(self, fits):
        chi2, p = mundlak_test(fits["cre"], fits["mundlak_bars"])
        assert chi2 >= 0
        assert 0 <= p <= 1

    def test_wykrywa_skorelowany_efekt_kraju(self, panel_skorelowany_efekt):
        """Gdy zamożność jest skorelowana z efektem kraju, test musi odrzucić H0."""
        cre, _, bars = cre_poisson(panel_skorelowany_efekt)
        _, p = mundlak_test(cre, bars)
        assert p < 0.05

    def test_srednie_mundlaka_sa_dodane_do_ramki(self, fits):
        bars = fits["mundlak_bars"]
        assert bars == [t + "_bar" for t in TERMS]
        assert all(b in fits["cre_df"].columns for b in bars)


class TestFitAll:
    def test_zwraca_wszystkie_specyfikacje(self, fits):
        for key in ("pooled", "pooled_cluster", "negbin", "fe", "pa", "cre"):
            assert fits[key] is not None, f"brak oszacowania: {key}"

    def test_fe_pochlania_efekt_kraju(self, panel_skorelowany_efekt):
        """Przy zamożności skorelowanej z efektem kraju pooled przeszacowuje PKB.

        To jest mechanizm, na którym stoi główny wniosek projektu: po dodaniu
        efektów stałych kraju współczynnik przy PKB musi zmaleć.
        """
        fits = fit_all(panel_skorelowany_efekt)
        pooled = fits["pooled_cluster"].params["log_gdp_pc"]
        fe = fits["fe"].coef()["log_gdp_pc"]
        assert fe < pooled

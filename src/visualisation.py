"""Wykresy i tabele wynikowe. Funkcje bezstanowe, katalog docelowy globalny.

Wykresy powstaja z przekazanych oszacowan, nie z liczb wpisanych w kod -
inaczej rysunek rozjezdza sie z modelem przy pierwszej zmianie specyfikacji.

Barwy pochodza wylacznie z tokenow modulu `plot_style` i sa wybierane przez
role, jaka seria pelni w danych: cynober to przedmiot uwagi, blekit to
odniesienie. Zamoznosc jest przedmiotem pytania badawczego, wiec niesie
ACCENT; populacja jest zmienna towarzyszaca i zostaje przy REFERENCE.
"""

import numpy as np
import pandas as pd

from src import plot_style as style
from src.data_load import PROJECT_ROOT
from src.inference import LABELS

FIGURES = PROJECT_ROOT / "reports" / "figures"
TABLES = PROJECT_ROOT / "reports" / "tables"

# Obciecie ogona na histogramie: pojedyncze kraje z kilkudziesiecioma medalami
# rozciagalyby os tak, ze masa rozkladu przy zerze bylaby nieczytelna.
HIST_CLIP = 50


def setup_style() -> None:
    """Ustawia system wizualny projektu. Wywolac raz, przed pierwszym rysunkiem."""
    style.apply_style()


def _finish(fig, name: str | None, show: bool):
    """Zapis do reports/figures i/lub wyswietlenie; zamyka figure po zapisie."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if name:
        FIGURES.mkdir(parents=True, exist_ok=True)
        path = FIGURES / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        print("Zapisano wykres:", path.name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_medal_distribution(df: pd.DataFrame, name="rozklad_medali", show=True):
    """Rozklad liczby medali - skosnosc i udzial zer.

    Os liczebnosci w skali logarytmicznej, bo inaczej slupek przy zerze
    (ponad polowa obserwacji) przykrywa caly ogon.
    """
    import matplotlib.pyplot as plt

    share_zero = (df["medals"] == 0).mean()
    fig, ax = plt.subplots(figsize=style.SMALL)
    ax.hist(
        df["medals"].clip(upper=HIST_CLIP),
        bins=range(0, HIST_CLIP + 2),
        color=style.REFERENCE,
        edgecolor=style.BACKGROUND,
        linewidth=0.4,
    )
    ax.set_xlabel(f"Liczba medali na jednych igrzyskach (obcięte do {HIST_CLIP})")
    ax.set_ylabel("Liczba obserwacji (kraj × igrzyska)")
    udzial = f"{share_zero:.1%}".replace(".", ",")
    ax.set_title(f"Rozkład liczby medali (udział zer = {udzial})")
    ax.set_yscale("log")
    return _finish(fig, name, show)


def plot_gdp_effect(comp: pd.DataFrame, name="efekt_pkb_estymatory", show=True):
    """Przedzialy ufnosci efektu PKB w kolejnych estymatorach.

    `comp` pochodzi z `inference.compare_gdp_effect`, wiec wykres zmienia sie
    razem z modelami.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SMALL)
    ax.errorbar(
        comp["coef_log_gdp_pc"],
        comp["model"],
        xerr=1.96 * comp["se"],
        fmt="o",
        capsize=4,
        color=style.REFERENCE,
        ecolor=style.MUTED,
    )
    ax.axvline(0, color=style.AXES, ls="--", lw=1)
    ax.set_xlabel("Współczynnik przy log_gdp_pc (95% CI)")
    ax.set_title("Efekt PKB per capita: pooled vs PA vs FE vs CRE")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.decimal_comma(ax, axis="x", places=1)
    return _finish(fig, name, show)


def plot_medals_vs_covariates(df: pd.DataFrame, name="medale_pkb_populacja", show=True):
    """Surowe zaleznosci medali od zamoznosci i populacji.

    Przekroj ograniczony do obserwacji z co najmniej jednym medalem: zera
    tworzylyby linie na dole, ktora nie mowi nic o sile zaleznosci.
    """
    import matplotlib.pyplot as plt

    sub = df[df["medals"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=style.STANDARD, sharey=True)
    for ax, col, colour, title in (
        (axes[0], "log_gdp_pc", style.ACCENT, "Medale a zamożność"),
        (axes[1], "log_pop", style.REFERENCE, "Medale a populacja"),
    ):
        ax.scatter(sub[col], np.log1p(sub["medals"]), s=8, alpha=0.3, color=colour)
        ax.set_xlabel("log PKB per capita" if col == "log_gdp_pc" else "log populacji")
        ax.set_title(title)
        style.decimal_comma(ax, axis="x", places=1)
    axes[0].set_ylabel("log(1+medale)")
    style.decimal_comma(axes[0], axis="y", places=1)
    return _finish(fig, name, show)


# --- Tabele publikacyjne -------------------------------------------------

CSS = """<style>
table.sci { border-collapse: collapse; margin: 8px 0 4px 0; font-family:'Times New Roman',Times,serif; font-size:11pt; color:#000; }
table.sci th, table.sci td { padding:3px 12px; text-align:center; border:none; }
table.sci th { border-top:2px solid #000; border-bottom:1px solid #000; font-weight:bold; }
table.sci tr:last-child td { border-bottom:2px solid #000; }
table.sci td:first-child, table.sci th:first-child { text-align:left; }
.sci .sub td { padding-top:0; color:#333; }
.sci .div td { border-top:1px solid #000; font-weight:bold; text-align:left; padding-top:10px; padding-bottom:4px; }
.note { font-family:'Times New Roman',Times,serif; font-size:9pt; max-width:760px; margin:4px 0 20px 0; text-align:left; }
</style>"""

SOURCE_NOTE = "<em>Źródło:</em> Opracowanie własne."


def stars(p: float) -> str:
    """Gwiazdki istotnosci: *** 1%, ** 5%, * 10%."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def fmt(x: float) -> str:
    """Polski separator dziesietny i minus typograficzny."""
    return f"{x:.3f}".replace("-", "−").replace(".", ",")


def _has(d: dict, key: str) -> bool:
    v = d.get(key)
    return key in d and not (isinstance(v, float) and np.isnan(v))


def save_table(html: str, fname: str, show: bool = True) -> str:
    """Zapis tabeli do reports/tables i wyswietlenie w notebooku."""
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / fname
    path.write_text(
        "<!DOCTYPE html><html lang='pl'><head><meta charset='UTF-8'></head><body>"
        + html
        + "</body></html>",
        encoding="utf-8",
    )
    print("Zapisano tabelę:", path.name)
    if show:
        try:
            from IPython.display import HTML, display

            display(HTML(html))
        except ImportError:
            pass
    return html


def reg_table_html(title, terms, coef, se, pval, stats, note, labels=None) -> str:
    """Tabela jednego modelu: Zmienna | Wspolczynnik (SE pod spodem) | IRR."""
    labels = labels or LABELS
    body = ""
    for t in terms:
        b, s, p = coef[t], se[t], pval[t]
        body += (
            f"<tr class='main'><td><strong>{labels.get(t, t)}</strong></td>"
            f"<td>{fmt(b)}{stars(p)}</td><td>{fmt(np.exp(b))}</td></tr>"
            f"<tr class='sub'><td></td><td>({fmt(s)})</td><td></td></tr>"
        )
    body += "<tr class='div'><td colspan='3'>Statystyki i diagnostyka modelu</td></tr>"
    for k, v in stats:
        body += f"<tr><td>{k}</td><td colspan='2'>{v}</td></tr>"
    return (
        CSS + f"<p><strong>{title}</strong></p>"
        "<table class='sci'><tr><th>Zmienna</th><th>Współczynnik</th><th>IRR</th></tr>"
        + body
        + "</table>"
        + f"<div class='note'>{SOURCE_NOTE} {note}</div>"
    )


def multi_table_html(title, models, row_keys, stat_rows, note) -> str:
    """Tabela zbiorcza: kolumna na model, wiersz na zmienna.

    `models` to lista (nazwa, coef, se, pval); wiersz pomijany, jesli zaden
    model nie szacuje danej zmiennej.
    """
    names = [m[0] for m in models]
    ncol = len(names) + 1
    head = "<tr><th>Zmienna</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    body = ""
    for key, lab in row_keys:
        if not any(_has(m[1], key) for m in models):
            continue
        main = sub = ""
        for _, c, s, p in models:
            if _has(c, key):
                main += f"<td>{fmt(c[key])}{stars(p[key])}</td>"
                sub += f"<td>({fmt(s[key])})</td>"
            else:
                main += "<td>—</td>"
                sub += "<td></td>"
        body += f"<tr class='main'><td><strong>{lab}</strong></td>{main}</tr>"
        body += f"<tr class='sub'><td></td>{sub}</tr>"
    body += f"<tr class='div'><td colspan='{ncol}'>Statystyki i diagnostyka modelu</td></tr>"
    for lab, vals in stat_rows:
        body += f"<tr><td>{lab}</td>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
    return (
        CSS
        + f"<p><strong>{title}</strong></p><table class='sci'>"
        + head
        + body
        + "</table>"
        + f"<div class='note'>{SOURCE_NOTE} {note}</div>"
    )


def simple_table_html(title, header, rows, note) -> str:
    """Tabela bez par wspolczynnik/SE - dopasowanie modeli, porownania."""
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
    body = "".join(
        "<tr class='main'>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        CSS
        + f"<p><strong>{title}</strong></p><table class='sci'>"
        + head
        + body
        + "</table>"
        + f"<div class='note'>{SOURCE_NOTE} {note}</div>"
    )

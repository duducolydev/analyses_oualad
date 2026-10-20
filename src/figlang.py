# -*- coding: utf-8 -*-
"""Langue des figures : français (thèse, figures/) ou anglais (article, figures/en/)."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

TEXTE, TEXTE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"


def langue(description: str) -> tuple[str, Path]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--lang", choices=["fr", "en"], default="fr")
    lang = ap.parse_args().lang
    dossier = Path("figures") if lang == "fr" else Path("figures") / "en"
    dossier.mkdir(parents=True, exist_ok=True)
    return lang, dossier


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": TEXTE2, "axes.labelcolor": TEXTE, "xtick.color": TEXTE2,
        "ytick.color": TEXTE2, "text.color": TEXTE, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })


def enregistrer(fig, dossier: Path, nom: str):
    for ext in ["pdf", "png"]:
        fig.savefig(dossier / f"{nom}.{ext}", dpi=200)
    print(f"Écrit : {dossier / nom}.pdf, .png")

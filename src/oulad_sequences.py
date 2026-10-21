# -*- coding: utf-8 -*-
"""
Variables SÉQUENTIELLES (T5) : activité VLE semaine par semaine jusqu'au cutoff.

K = nombre de semaines complètes observées dans TOUTES les présentations au
cutoff, K = floor((min_p d_c + 1) / 7) : les séquences ont la même longueur et
ne couvrent que des jours <= d_c (semaine w = jours 7w à 7w+6, w = 0..K-1).

Colonnes, alignées sur les lignes de construire() :
  seq_cWW  clics de la semaine WW          seq_jWW  jours actifs de la semaine WW
  pre_clics            clics avant le début du cours (date < 0)
  tend_pente           pente de log(1 + clics) sur les K semaines
  tend_part_fin        part des clics des K semaines tombant dans la seconde moitié
  tend_sem_actives     nombre de semaines avec au moins un clic
  tend_sem_inactives   semaines écoulées depuis la dernière semaine active (K si aucune)
relatives=True : chaque colonne est remplacée par son rang centile dans la
présentation, comme les variables relatives agrégées (D7).
"""
import numpy as np
import pandas as pd

from oulad_features import CLE


def n_semaines(courses: pd.DataFrame, cutoff: float) -> int:
    d_c = (courses["module_presentation_length"] * cutoff).round().astype(int)
    return int((d_c.min() + 1) // 7)


def sequences(tables: dict[str, pd.DataFrame], cles: pd.DataFrame, cutoff: float,
              relatives: bool = False) -> pd.DataFrame:
    """`cles` : colonnes code_module, code_presentation, id_student (ordre conservé)."""
    k = n_semaines(tables["courses"], cutoff)
    cle = CLE + ["id_student"]
    vle = tables["studentVle"]
    vle = vle[vle["date"] <= 7 * k - 1]

    pre = vle[vle["date"] < 0].groupby(cle)["sum_click"].sum().rename("pre_clics")
    sem = vle[vle["date"] >= 0].assign(semaine=lambda d: d["date"] // 7)
    clics = sem.groupby(cle + ["semaine"])["sum_click"].sum().unstack(fill_value=0)
    jours = sem.groupby(cle + ["semaine"])["date"].nunique().unstack(fill_value=0)
    clics = clics.reindex(columns=range(k), fill_value=0)
    jours = jours.reindex(columns=range(k), fill_value=0)
    clics.columns = [f"seq_c{w:02d}" for w in range(k)]
    jours.columns = [f"seq_j{w:02d}" for w in range(k)]

    out = cles[cle].merge(clics.join(jours, how="outer").join(pre, how="outer").reset_index(),
                          on=cle, how="left").fillna(0)
    c = out[[f"seq_c{w:02d}" for w in range(k)]].to_numpy(dtype=float)

    # Tendances
    y = np.log1p(c)
    t = np.arange(k) - (k - 1) / 2
    out["tend_pente"] = (y * t).sum(axis=1) / max((t ** 2).sum(), 1e-9)
    total = c.sum(axis=1)
    out["tend_part_fin"] = np.where(total > 0, c[:, k - k // 2:].sum(axis=1) / np.maximum(total, 1e-9), 0.0)
    actif = c > 0
    out["tend_sem_actives"] = actif.sum(axis=1)
    derniere = np.where(actif.any(axis=1), k - 1 - np.argmax(actif[:, ::-1], axis=1), -1)
    out["tend_sem_inactives"] = k - 1 - derniere

    colonnes = [x for x in out.columns if x not in cle]
    if relatives:
        out[colonnes] = out.groupby(CLE)[colonnes].rank(pct=True, method="average")
    return out[colonnes].reset_index(drop=True)

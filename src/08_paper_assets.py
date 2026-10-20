#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T7 — Chiffres et tableaux de l'article, générés depuis results/ et les données.

Écrit dans paper/generated/ :
  macros.tex            chaque nombre cité dans le texte (\\resXxx), pour qu'aucun
                        chiffre ne soit recopié à la main ;
  tab_modules.tex       caractéristiques des modules (cutoff 25 %, population active)
  tab_audit.tex         audit méthodologique : effet cumulé des corrections
  tab_transfer.tex      transfert par module : classement et signalement
  tab_remedies.tex      dérive et précision sous les deux remèdes
  tab_correlations.tex  corrélations de Spearman retenues (q BH < 0,10)
  tab_fairness.tex      écarts d'équité avec IC 95 %
  tab_target.tex        robustesse à la définition de la cible

Usage : python src/08_paper_assets.py
"""
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oulad_features import CLE, charger, construire

RES, OUT = Path("results"), Path("paper/generated")
META = "_c25_active"
MACROS: dict[str, str] = {}
METRIQUES = ["AUC", "Rappel", "Précision", "F1"]


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def signe(x, d=3):
    return f"{x:+.{d}f}".replace("-", "$-$")


def m(nom, val):
    assert re.fullmatch(r"res[A-Za-z]+", nom), nom
    assert nom not in MACROS, nom
    MACROS[nom] = val


def ecrire(nom, contenu):
    (OUT / nom).write_text(contenu, encoding="utf-8")


def moy_modules(fichier, cutoff=0.25, population="active"):
    d = pd.read_csv(RES / fichier)
    d = d[(d.cutoff == cutoff) & (d.population == population)]
    return d.groupby("protocole")[METRIQUES].mean()


# ---------------------------------------------------------------------------
def audit_et_donnees(tables):
    info, reg, co = tables["studentInfo"], tables["studentRegistration"], tables["courses"]
    m("resNEnrol", f"{len(info):,}")
    m("resNModules", str(info.code_module.nunique()))
    m("resNPres", str(co.groupby(CLE).ngroups))
    m("resRiskAll", pct(info.final_result.isin(["Withdrawn", "Fail"]).mean()))

    r = reg.merge(co, on=CLE).merge(info[CLE + ["id_student", "final_result"]], on=CLE + ["id_student"])
    cd = (r.module_presentation_length * 0.25).round()
    parti = r.date_unregistration.notna() & (r.date_unregistration <= cd)
    m("resPctGone", pct(parti.mean()))
    m("resPctWithdrawnGone", pct(parti.sum() / (r.final_result == "Withdrawn").sum()))
    par_mod = parti.groupby(r.code_module).mean()
    m("resGoneMin", pct(par_mod.min()))
    m("resGoneMinMod", par_mod.idxmin())
    m("resGoneMax", pct(par_mod.max()))
    m("resGoneMaxMod", par_mod.idxmax())
    m("resNGoneBeforeStart", f"{(r.date_unregistration < 0).sum():,}")

    a = tables["assessments"].merge(co, on=CLE)
    a["cutoff_day"] = (a.module_presentation_length * 0.25).round()
    a = a[a.date.fillna(1e9) <= a.cutoff_day]
    s = tables["studentAssessment"].merge(a[["id_assessment", "cutoff_day"]], on="id_assessment")
    m("resPctLateSub", pct((s.date_submitted > s.cutoff_day).mean()))

    f = construire(tables, 0.25, "active")
    m("resNActive", f"{len(f):,}")
    m("resRiskActive", pct(f.at_risk.mean()))
    g = f.code_module == "GGG"
    med = f.loc[g, "total_clicks"].median()
    m("resGGGMedClicks", f"{med:.0f}")
    m("resOthersMedClicks", f"{f.loc[~g, 'total_clicks'].median():.0f}")
    m("resGGGClickPctile", f"{100 * (f.loc[~g, 'total_clicks'] < med).mean():.0f}")
    m("resRiskLowClicksOthers", pct(f.loc[~g & (f.total_clicks <= med), "at_risk"].mean(), 0))
    m("resRiskLowClicksGGG", pct(f.loc[g & (f.total_clicks <= med), "at_risk"].mean(), 0))
    note = f.n_submissions > 0
    m("resGGGNoScore", pct(1 - note[g].mean(), 0))
    m("resRiskNoScoreOthers", pct(f.loc[~g & ~note, "at_risk"].mean(), 0))
    m("resRiskNoScoreGGG", pct(f.loc[g & ~note, "at_risk"].mean(), 0))
    sc = f.mean_early_score.where(f.mean_early_score >= 0)
    m("resGGGMedScore", f"{sc[g].median():.0f}")
    m("resOthersMedScore", f"{sc[~g].median():.0f}")

    # Dérive GGG à d'autres cutoffs (persistance)
    for c, nom in [(33, "resGGGDriftThirtyThree"), (50, "resGGGDriftFifty")]:
        p = pd.read_parquet(RES / "preds" / f"preds_rf_active_c{c}.parquet")
        p = p[p.code_module == "GGG"]
        m(nom, signe((p.p_lomo >= 0.5).mean() - (p.p_aleatoire >= 0.5).mean(), 2))


def tableau_audit():
    txt = (RES / "reference_rf_c25.txt").read_text(encoding="utf-8")
    ref_a = float(re.search(r"\(A\).*?\nAUC\s+([\d.]+)", txt, re.S).group(1))
    ref_b = float(re.search(r"Moyenne LOMO :\nAUC\s+([\d.]+)", txt).group(1))
    ref_gap = float(re.search(r"cart d'AUC \(aléatoire - LOMO\) : ([+-][\d.]+)", txt).group(1))
    abso = pd.read_csv(RES / "early_curve.csv")
    abso = abso[abso.cutoff == 0.25].set_index(["population", "protocole"])["AUC"]
    m_act = moy_modules("early_curve_by_module.csv")["AUC"]
    m_rel = moy_modules("early_curve_by_module_relatives.csv")["AUC"]
    lignes = [
        ("Reference protocol (all enrolments, pooled AUC in (A))", ref_a, ref_b),
        ("+ early scores filtered on submission date", abso[("all", "aleatoire")], abso[("all", "lomo")]),
        ("+ students still enrolled at the cut-off only", abso[("active", "aleatoire")], abso[("active", "lomo")]),
        ("+ same aggregation in (A) and (B)", m_act["aleatoire"], m_act["lomo"]),
        ("+ context-relative features", m_rel["aleatoire"], m_rel["lomo"]),
    ]
    m("resRefA", f3(ref_a))
    m("resRefB", f3(ref_b))
    m("resRefGap", f3(ref_gap))
    m("resPooledActiveA", f3(abso[("active", "aleatoire")]))
    m("resPooledGap", f3(abso[("active", "aleatoire")] - abso[("active", "lomo")]))
    m("resSameAggGap", f3(m_act["aleatoire"] - m_act["lomo"]))
    # écart de la ligne 1 : valeur non arrondie de la sortie de référence
    ecarts = [ref_gap] + [a - b for _, a, b in lignes[1:]]
    corps = "\n".join(f"{i}) {lib} & {f3(a)} & {f3(b)} & {f3(e)} \\\\"
                      for i, ((lib, a, b), e) in enumerate(zip(lignes, ecarts), 1))
    ecrire("tab_audit.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Methodological audit at the 25\% cut-off (random forest). Each row adds one correction to the previous one. AUC of protocol (A), random split grouped by student, and (B), leave-one-module-out (LOMO), mean over held-out modules.}}
\label{{tab:audit}}
\begin{{tabular}}{{@{{}}p{{4.3cm}}ccc@{{}}}}
\toprule
Evaluation setting & AUC (A) & AUC (B) & Gap \\
\midrule
{corps}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def courbe():
    for suf, var in [("", "Abs"), ("_relatives", "Rel")]:
        d = pd.read_csv(RES / f"early_curve_by_module{suf}.csv")
        d = d[d.population == "active"].groupby(["protocole", "cutoff"])[METRIQUES].mean()
        for c, cn in [(0.10, "Ten"), (0.25, "TwentyFive"), (0.50, "Fifty")]:
            m(f"resAuc{var}A{cn}", f3(d.loc[("aleatoire", c), "AUC"]))
            m(f"resAuc{var}B{cn}", f3(d.loc[("lomo", c), "AUC"]))
        for c, cn in [(0.25, "TwentyFive")]:
            for met, mn in [("Rappel", "Rec"), ("Précision", "Prec"), ("F1", "Fone")]:
                m(f"res{mn}{var}A{cn}", f3(d.loc[("aleatoire", c), met]))
                m(f"res{mn}{var}B{cn}", f3(d.loc[("lomo", c), met]))
        gaps = [d.loc[("aleatoire", c), "AUC"] - d.loc[("lomo", c), "AUC"]
                for c in sorted(d.index.get_level_values(1).unique())]
        m(f"resGap{var}Min", f3(min(gaps)))
        m(f"resGap{var}Max", f3(max(gaps)))


def tableau_modules(t):
    cols = ["n", "taux_risque", "part_withdrawn_parmi_risque", "premiere_echeance", "part_examen",
            "part_note_precoce", "clics_medians", "part_plus35"]
    corps = []
    for mod, r in t[cols].iterrows():
        corps.append(f"{mod} & {int(r.n):,} & {pct(r.taux_risque)} & {pct(r.part_withdrawn_parmi_risque)} & "
                     f"{f2(r.premiere_echeance)} & {pct(r.part_examen, 0)} & {pct(r.part_note_precoce)} & "
                     f"{r.clics_medians:.0f} & {pct(r.part_plus35)} \\\\")
    ecrire("tab_modules.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Module characteristics at the 25\% cut-off, students still enrolled at the cut-off. At-risk: Withdrawn or Fail. First deadline: first non-exam assessment, as a fraction of the presentation length. Exam weight: share of the exam in the total assessment weight. Early score: share of students with at least one assessment submitted before the cut-off. Values are averaged over the presentations of each module.}}
\label{{tab:modules}}
\footnotesize
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
 & & At-risk & Withdrawn among & First & Exam & Early & Median & Aged \\
Module & $n$ & (\%) & at-risk (\%) & deadline & weight (\%) & score (\%) & clicks & 35+ (\%) \\
\midrule
{chr(10).join(corps)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def transfert(t, tr):
    m("resCostAbs", f3(t.cout_transfert_auc.mean()))
    m("resCostRel", f3(tr.cout_transfert_auc.mean()))
    m("resDriftAbs", f3(t.derive_signalement.abs().mean()))
    m("resDriftRel", f3(tr.derive_signalement.abs().mean()))
    g, gr = t.loc["GGG"], tr.loc["GGG"]
    m("resGGGFlagA", pct(g.signal_aleatoire, 0))
    m("resGGGFlagB", pct(g.signal_lomo, 0))
    m("resGGGRisk", pct(g.taux_risque, 0))
    m("resGGGDrift", signe(g.derive_signalement, 2))
    m("resGGGDriftRel", signe(gr.derive_signalement, 2))
    m("resGGGRecA", f2(g.rappel_aleatoire))
    m("resGGGRecB", f2(g.rappel_lomo))
    m("resGGGPrecA", f2(g.precision_aleatoire))
    m("resGGGPrecB", f2(g.precision_lomo))
    m("resGGGAucA", f3(g.auc_aleatoire))
    m("resAAAAucA", f3(t.loc["AAA"].auc_aleatoire))
    m("resBBBDrift", signe(t.loc["BBB"].derive_signalement, 2))
    m("resBBBCost", f3(t.loc["BBB"].cout_transfert_auc))
    corps = []
    for mod, r in t.iterrows():
        q = tr.loc[mod]
        corps.append(f"{mod} & {f3(r.auc_aleatoire)} & {f3(r.auc_lomo)} & {signe(r.cout_transfert_auc)} & "
                     f"{pct(r.taux_risque, 0)} & {pct(r.signal_aleatoire, 0)} & {pct(r.signal_lomo, 0)} & "
                     f"{f3(q.auc_lomo)} & {pct(q.signal_lomo, 0)} \\\\")
    moy = (f"Mean & {f3(t.auc_aleatoire.mean())} & {f3(t.auc_lomo.mean())} & {signe(t.cout_transfert_auc.mean())} & "
           f"{pct(t.taux_risque.mean(), 0)} & {pct(t.signal_aleatoire.mean(), 0)} & {pct(t.signal_lomo.mean(), 0)} & "
           f"{f3(tr.auc_lomo.mean())} & {pct(tr.signal_lomo.mean(), 0)} \\\\")
    ecrire("tab_transfer.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Ranking versus threshold transfer per module (25\% cut-off, students still enrolled). AUC and flag rate ($p \geq 0.5$) are computed on the students of each module; (A) uses out-of-fold predictions of the random split, (B) the model trained on the six other modules.}}
\label{{tab:transfer}}
\begin{{tabular}}{{@{{}}lccccccccc@{{}}}}
\toprule
 & \multicolumn{{3}}{{c}}{{Ranking (absolute features)}} & \multicolumn{{3}}{{c}}{{Flagged students, \% (absolute)}} & \multicolumn{{2}}{{c}}{{Relative features, (B)}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(l){{8-9}}
Module & AUC (A) & AUC (B) & Cost & Observed risk & (A) & (B) & AUC & Flagged (\%) \\
\midrule
{chr(10).join(corps)}
\midrule
{moy}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def remedes(t, tr):
    rk = pd.read_csv(RES / f"recalibration_k{META}.csv", header=[0, 1], index_col=[0, 1, 2])
    mk = rk.xs("mean", axis=1, level=1).xs(100, level="k")
    rt = pd.read_csv(RES / f"recalibration_temporelle{META}.csv").set_index(["module", "methode"])
    derive_k = mk["signalement"].unstack()["lomo_recalibre"] - mk["signalement"].unstack()["aleatoire"]
    derive_t = (rt["signalement"].unstack()["lomo_recalibre"] - rt["signalement"].unstack()["aleatoire"])
    m("resDriftRecalK", f3(derive_k.abs().mean()))
    m("resDriftRecalT", f3(derive_t.abs().mean()))
    m("resGGGDriftRecalK", signe(derive_k["GGG"], 2))
    m("resGGGPrecRecalK", f2(mk.loc["GGG", "precision"]["lomo_recalibre"]))
    m("resGGGPrecRecalT", f2(rt.loc[("GGG", "lomo_recalibre"), "precision"]))
    m("resGGGRecRecalT", f2(rt.loc[("GGG", "lomo_recalibre"), "rappel"]))
    m("resFoneRecalK", f3(mk["f1"].unstack()["lomo_recalibre"].mean()))
    m("resFoneRawK", f3(mk["f1"].unstack()["lomo_brut"].mean()))
    corps = []
    for mod in t.index:
        corps.append(f"{mod} & {signe(t.loc[mod, 'derive_signalement'], 2)} & {signe(derive_k[mod], 2)} & "
                     f"{signe(derive_t[mod], 2)} & {signe(tr.loc[mod, 'derive_signalement'], 2)} & "
                     f"{f2(t.loc[mod, 'precision_lomo'])} & {f2(mk.loc[mod, 'precision']['lomo_recalibre'])} & "
                     f"{f2(rt.loc[(mod, 'lomo_recalibre'), 'precision'])} & {f2(tr.loc[mod, 'precision_lomo'])} \\\\")
    moy = (f"Mean $|\\cdot|$ / mean & {f2(t.derive_signalement.abs().mean())} & {f2(derive_k.abs().mean())} & "
           f"{f2(derive_t.abs().mean())} & {f2(tr.derive_signalement.abs().mean())} & "
           f"{f2(t.precision_lomo.mean())} & {f2(mk['precision'].unstack()['lomo_recalibre'].mean())} & "
           f"{f2(rt.xs('lomo_recalibre', level='methode')['precision'].mean())} & {f2(tr.precision_lomo.mean())} \\\\")
    ecrire("tab_remedies.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Flag-rate drift (B) $-$ (A) and precision of the transferred model under the two remedies (25\% cut-off). Platt-$k$: Platt recalibration on $k=100$ randomly drawn labelled students of the target module (mean of 50 draws, evaluated on the remaining students). Platt-pilot: recalibration on the first presentation of the module, evaluated on the later ones (drift measured against (A) on the same students). Relative: context-relative features, no labels.}}
\label{{tab:remedies}}
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{Flag-rate drift}} & \multicolumn{{4}}{{c}}{{Precision of (B)}} \\
\cmidrule(lr){{2-5}}\cmidrule(l){{6-9}}
Module & Raw & Platt-$k$ & Platt-pilot & Relative & Raw & Platt-$k$ & Platt-pilot & Relative \\
\midrule
{chr(10).join(corps)}
\midrule
{moy}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def correlations():
    c = pd.read_csv(RES / f"transfer_correlations{META}.csv")
    noms = {"part_plus35": "Share aged 35+", "part_withdrawn_parmi_risque": "Withdrawn share among at-risk",
            "taux_risque": "At-risk rate", "clics_medians": "Median clicks", "part_sans_alevel": "Share without A-level",
            "ecart_structure_eval": "Assessment-structure distance", "premiere_echeance": "First deadline",
            "part_note_precoce": "Share with early score", "n_types_activite": "VLE activity types",
            "n_eval_avant_cutoff": "Assessments due before cut-off"}
    axes = {"auc_aleatoire": "Difficulty", "cout_transfert_auc": "Rank. cost",
            "derive_signalement": "Drift"}
    s = c[c.q_valeur_BH < 0.10].sort_values(["niveau", "axe", "p_valeur"])
    r = c[(c.niveau == "module") & (c.descripteur == "part_plus35") & (c.axe == "auc_aleatoire")].iloc[0]
    m("resRhoAgeDifficulty", signe(r.spearman_rho, 2))
    r = c[(c.niveau == "presentation") & (c.descripteur == "part_withdrawn_parmi_risque")
          & (c.axe == "derive_signalement")].iloc[0]
    m("resRhoWithdrawnDrift", signe(r.spearman_rho, 2))
    r = c[(c.niveau == "presentation") & (c.descripteur == "part_plus35") & (c.axe == "auc_aleatoire")].iloc[0]
    m("resRhoAgeDifficultyPres", signe(r.spearman_rho, 2))
    corps = [f"{'Mod.' if r.niveau == 'module' else 'Pres.'} ({r.n_contextes}) & {axes[r.axe]} & "
             f"{noms.get(r.descripteur, r.descripteur)} & {signe(r.spearman_rho, 2)} & {r.q_valeur_BH:.3f} \\\\"
             for r in s.itertuples()]
    m("resNCorrTests", str(len(c)))
    ecrire("tab_correlations.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Context descriptors associated with the three transfer axes (Spearman's $\rho$, Benjamini--Hochberg $q < 0.10$ within level, {len(c)} tests in total). Mod.: modules; Pres.: presentations, which are not independent (they share the module-level LOMO model). Difficulty: AUC (A); Rank.\ cost: AUC (A) $-$ AUC (B); Drift: flag-rate drift. Exploratory associations.}}
\label{{tab:corr}}
\footnotesize
\begin{{tabular}}{{@{{}}llp{{3.3cm}}rr@{{}}}}
\toprule
Level ($n$) & Axis & Descriptor & $\rho$ & $q$ \\
\midrule
{chr(10).join(corps)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def shap():
    st = pd.read_csv(RES / f"shap_stability{META}.csv").set_index(["variables", "module"])
    c = st["spearman_transfere_local"].unstack(0)
    m("resConcAbs", f2(c["absolues"].mean()))
    m("resConcRel", f2(c["relatives"].mean()))
    m("resConcGGGAbs", f2(c.loc["GGG", "absolues"]))
    m("resConcGGGRel", f2(c.loc["GGG", "relatives"]))
    m("resNConcImproved", str(int((c["relatives"] > c["absolues"]).sum())))
    t5 = st["top5_communs"].unstack(0)
    m("resTopFiveAbs", f"{t5['absolues'].mean():.1f}")
    m("resTopFiveRel", f"{t5['relatives'].mean():.1f}")
    for v, vn in [("absolues", "Abs"), ("relatives", "Rel")]:
        imp = pd.read_csv(RES / f"shap_by_module{META}_{v}.csv")
        loc = imp[imp.modele == "local"].pivot(index="variable", columns="code_module", values="mean_abs_shap")
        paires = [spearmanr(loc[a], loc[b])[0] for a, b in combinations(loc.columns, 2)]
        m(f"resStab{vn}", f2(float(np.mean(paires))))
    d = pd.read_csv(RES / f"shap_drift_decomposition{META}.csv").set_index(["variables", "code_module"])
    g = d.loc[("absolues", "GGG")]
    m("resGGGShapTotal", signe(g["total"], 2))
    m("resGGGShapScores", signe(g["Notes précoces"], 2))
    m("resGGGShapEngage", signe(g["Engagement VLE"], 2))
    m("resGGGShapDemo", signe(g["Démographie, inscription"], 2))
    m("resGGGShapTotalRel", signe(d.loc[("relatives", "GGG"), "total"], 2))
    tm = pd.read_csv(RES / f"transfer_by_module{META}.csv", index_col=0)
    rho = spearmanr(d.loc["absolues"]["total"], tm.loc[d.loc["absolues"].index, "derive_signalement"])[0]
    m("resRhoShapDrift", f2(rho))


def equite():
    g = pd.read_csv(RES / f"fairness_gaps{META}.csv")
    ch = pd.read_csv(RES / f"fairness_gap_changes{META}.csv")

    def ic(v, p, a, met):
        r = g[(g.variables == v) & (g.protocole == p) & (g.attribut == a) & (g.metrique == met)].iloc[0]
        return r

    for v, p, cn in [("absolues", "aleatoire", "AbsA"), ("absolues", "lomo", "AbsB"), ("relatives", "lomo", "RelB")]:
        r = ic(v, p, "genre", "FPR")
        m(f"resFprGender{cn}", signe(r.ecart))
        m(f"resFprGender{cn}CI", f"[{signe(r.ic_bas)}, {signe(r.ic_haut)}]")
        r = ic(v, p, "imd", "FPR")
        m(f"resFprImd{cn}", signe(r.ecart))
    r = ch[(ch.comparaison == "relatives − absolues, LOMO") & (ch.attribut == "imd") & (ch.metrique == "FPR")].iloc[0]
    m("resFprImdRelChange", signe(r.variation))
    m("resFprImdRelChangeCI", f"[{signe(r.ic_bas)}, {signe(r.ic_haut)}]")

    info = pd.read_parquet("data/cache/studentInfo.parquet")
    p = pd.read_parquet(RES / "preds" / "preds_rf_active_c25.parquet").merge(
        info[["code_module", "code_presentation", "id_student", "gender"]],
        on=["code_module", "code_presentation", "id_student"])
    partF = p.groupby("code_module").gender.apply(lambda s: (s == "F").mean())
    m("resShareFemaleBBB", pct(partF["BBB"], 0))
    m("resShareFemaleGGG", pct(partF["GGG"], 0))
    neg = p[p.at_risk == 0].assign(f=lambda d: d.p_lomo >= 0.5)
    t = neg.groupby(["code_module", "gender"]).f.mean().unstack()
    m("resFprGenderWithinMax", signe((t.F - t.M).max()))
    fr = pd.read_csv(RES / f"fairness_groups{META}.csv")
    q = fr[(fr.variables == "absolues") & (fr.protocole == "aleatoire") & (fr.attribut == "imd")].set_index("groupe")
    m("resRiskQOne", pct(q.loc["Q1", "taux_risque"], 0))
    m("resRiskQFive", pct(q.loc["Q5", "taux_risque"], 0))

    configs = [("absolues", "aleatoire", "(A) absolute"), ("absolues", "lomo", "(B) absolute"),
               ("relatives", "lomo", "(B) relative")]
    att = [("genre", "Female $-$ male"), ("imd", "IMD Q1 $-$ Q5"), ("handicap", "Disability yes $-$ no")]
    corps = []
    for a, lib in att:
        for i, (v, p_, cl) in enumerate(configs):
            cells = []
            for met in ["TPR", "FPR", "PPV", "signalement"]:
                r = ic(v, p_, a, met)
                cells.append(f"{signe(r.ecart)} [{signe(r.ic_bas)}, {signe(r.ic_haut)}]")
            tete = f"\\multirow{{3}}{{*}}{{{lib}}}" if i == 0 else ""
            corps.append(f"{tete} & {cl} & " + " & ".join(cells) + " \\\\")
        corps.append("\\midrule")
    corps = corps[:-1]
    ecrire("tab_fairness.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Between-group gaps in error rates at the 25\% cut-off (threshold 0.5), with 95\% confidence intervals from a student-level cluster bootstrap (1{{,}}000 resamples). TPR: recall; FPR: false-positive rate; PPV: precision; Flag: share of students flagged.}}
\label{{tab:fairness}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\toprule
Contrast & Configuration & TPR gap & FPR gap & PPV gap & Flag-rate gap \\
\midrule
{chr(10).join(corps)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def cible():
    lignes = []
    for c, cs, cl in [("withdrawn_fail", "", "W or F"), ("withdrawn", "_withdrawn", "W only")]:
        for v, vs, vl in [("absolues", "", "abs."), ("relatives", "_relatives", "rel.")]:
            t = pd.read_csv(RES / f"transfer_by_module{META}{vs}{cs}.csv", index_col=0)
            lignes.append((cl, vl, t.taux_risque.mean(), t.auc_aleatoire.mean(), t.auc_lomo.mean(),
                           t.cout_transfert_auc.mean(), t.derive_signalement.abs().mean(),
                           t.loc["GGG", "derive_signalement"]))
            if c == "withdrawn":
                n = "Abs" if v == "absolues" else "Rel"
                m(f"resWdAucB{n}", f3(t.auc_lomo.mean()))
                m(f"resWdAucA{n}", f3(t.auc_aleatoire.mean()))
                m(f"resWdCost{n}", f3(t.cout_transfert_auc.mean()))
                m(f"resWdDrift{n}", f3(t.derive_signalement.abs().mean()))
                m(f"resWdGGGDrift{n}", signe(t.loc["GGG", "derive_signalement"], 2))
                m(f"resWdRecB{n}", f2(t.rappel_lomo.mean()))
                if v == "absolues":
                    m("resWdRisk", pct(t.taux_risque.mean()))
    corps = "\n".join(f"{a} & {b} & {pct(r)} & {f3(x)} & {f3(y)} & {signe(z)} & {f3(w)} & {signe(gg, 2)} \\\\"
                      for a, b, r, x, y, z, w, gg in lignes)
    ecrire("tab_target.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Robustness to the target definition (25\% cut-off, students still enrolled; means over modules). W or F: Withdrawn or Fail; W only: Withdrawn only. Abs./rel.: absolute or relative features. Cost: AUC (A) $-$ AUC (B); Drift: mean absolute flag-rate drift.}}
\label{{tab:target}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}llcccccc@{{}}}}
\toprule
Target & Feat. & Risk (\%) & AUC (A) & AUC (B) & Cost & Drift & GGG \\
\midrule
{corps}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = charger(Path("data/OULAD"), Path("data/cache"))
    t = pd.read_csv(RES / f"transfer_by_module{META}.csv", index_col=0)
    tr = pd.read_csv(RES / f"transfer_by_module{META}_relatives.csv", index_col=0)
    audit_et_donnees(tables)
    tableau_audit()
    courbe()
    tableau_modules(t)
    transfert(t, tr)
    remedes(t, tr)
    correlations()
    shap()
    equite()
    cible()
    lignes = ["% Généré par src/08_paper_assets.py — ne pas modifier à la main."]
    lignes += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    ecrire("macros.tex", "\n".join(lignes) + "\n")
    print(f"{len(MACROS)} macros et 7 tableaux écrits dans {OUT}/")


if __name__ == "__main__":
    main()

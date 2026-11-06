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
    """Remèdes comparés sur les mêmes étudiants, contre une référence commune
    (modèle (A) à variables absolues) — 14_remedies.py, plans « k = 100 » et « pilote »."""
    rk = pd.read_csv(RES / "remedes_k_c25.csv", header=[0, 1], index_col=[0, 1, 2]).xs(100, level="k")
    mk, sk = rk.xs("mean", axis=1, level=1), rk.xs("std", axis=1, level=1)
    pil = pd.read_csv(RES / "remedes_pilote_c25.csv").set_index(["module", "regle"])
    d, prec = mk["derive_ref"].unstack("regle"), mk["precision"].unstack("regle")
    ecr, f1 = mk["ecart_risque"].unstack("regle"), mk["F1"].unstack("regle")
    sd = sk["derive_ref"].unstack("regle")
    dp = pil["derive_ref"].unstack("regle")
    m("resDriftRecalK", f3(d["platt"].abs().mean()))
    m("resDriftRecalT", f3(dp["platt"].abs().mean()))
    m("resDriftRawCommon", f3(d["brut"].abs().mean()))
    m("resDriftRelCommon", f3(d["relatives"].abs().mean()))
    m("resDriftTopQ", f3(d["top_q"].abs().mean()))
    m("resDriftRelPilot", f3(dp["relatives"].abs().mean()))
    m("resGGGDriftRecalK", signe(d.loc["GGG", "platt"], 2))
    m("resGGGDriftRelCommon", signe(d.loc["GGG", "relatives"], 2))
    m("resGGGPrecRecalK", f2(prec.loc["GGG", "platt"]))
    m("resGGGPrecRecalT", f2(pil.loc[("GGG", "platt"), "precision"]))
    m("resGGGRecRecalT", f2(pil.loc[("GGG", "platt"), "rappel"]))
    for regle, nom in [("brut", "Raw"), ("platt", "Recal"), ("top_q", "TopQ"), ("relatives", "Rel")]:
        m(f"resFone{nom}K", f3(f1[regle].mean()))
        m(f"resPrec{nom}K", f3(prec[regle].mean()))
        m(f"resPrec{nom}Pilot", f3(pil.xs(regle, level="regle")["precision"].mean()))
    for regle, nom in [("brut", "Raw"), ("platt", "Recal"), ("top_q", "TopQ"), ("top_q_oracle", "TopQOracle"),
                       ("relatives", "Rel"), ("reference_A", "RefA")]:
        m(f"resGapRisk{nom}", f3(ecr[regle].mean()))
    m("resPlattSdFlag", f3(sk["signalement"].unstack("regle")["platt"].mean()))

    regles = ["brut", "platt", "top_q", "relatives"]
    corps = []
    for mod in d.index:
        der = [signe(d.loc[mod, r], 2) + (f" ({sd.loc[mod, r]:.2f})" if r == "platt" else "") for r in regles]
        corps.append(f"{mod} & " + " & ".join(der) + " & " + " & ".join(f2(prec.loc[mod, r]) for r in regles) + " \\\\")
    moy = (r"Mean $|\cdot|$ / mean & " + " & ".join(f2(d[r].abs().mean()) for r in regles) + " & "
           + " & ".join(f2(prec[r].mean()) for r in regles) + " \\\\")
    risque = ("Mean $|$flagged $-$ risk$|$ & " + " & ".join(f2(ecr[r].mean()) for r in regles)
              + r" & \multicolumn{4}{c}{(A) absolute: " + f2(ecr["reference_A"].mean()) + "} \\\\")
    ecrire("tab_remedies.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Remedies compared on the same students and against the same reference (25\% cut-off). For each module, all rules are evaluated on the students not drawn for Platt recalibration ($k=100$ labelled students of the target module, mean of 50 draws; standard deviation across draws in parentheses). Drift: share flagged minus that of the absolute-feature model trained with the module seen, (A). Raw: absolute features, threshold 0.5. Top-$q$: in each cohort, the $q$ students with the highest scores, $q$ being the at-risk rate of the six training modules. Relative: context-relative features, threshold 0.5. Platt and the raw model use absolute features; only Platt uses labels of the target module.}}
\label{{tab:remedies}}
\footnotesize
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{Flag-rate drift vs (A)}} & \multicolumn{{4}}{{c}}{{Precision}} \\
\cmidrule(lr){{2-5}}\cmidrule(l){{6-9}}
Module & Raw & Platt-$k$ & Top-$q$ & Relative & Raw & Platt-$k$ & Top-$q$ & Relative \\
\midrule
{chr(10).join(corps)}
\midrule
{moy}
{risque}
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
    intra = pd.read_csv(RES / f"fairness_within_module{META}.csv")
    intra = intra[(intra.protocole == "lomo") & (intra.metrique == "FPR")].set_index(["variables", "attribut", "module"])
    ga = intra.xs(("absolues", "genre"), level=["variables", "attribut"])
    m("resFprGenderWithinMax", signe(ga.ecart.max()))
    m("resNGenderWithinNeg", str(int((ga.ic_haut < 0).sum())))
    m("resNGenderWithinPos", str(int((ga.ic_bas > 0).sum())))
    gr = intra.xs(("relatives", "genre"), level=["variables", "attribut"])
    m("resNGenderWithinPosRel", str(int((gr.ic_bas > 0).sum())))
    r = gr.loc["CCC"]
    m("resFprGenderCCCRel", signe(r.ecart))
    m("resFprGenderCCCRelCI", f"[{signe(r.ic_bas)}, {signe(r.ic_haut)}]")
    ia = intra.xs(("absolues", "imd"), level=["variables", "attribut"])
    m("resNImdWithinPos", str(int((ia.ic_bas > 0).sum())))
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


def sequence():
    """T5 : familles de modèles et information temporelle."""
    from fig_sequence import derive_moyenne, donnees
    d = donnees()
    auc = d.groupby(["variables", "modele", "cutoff", "protocole"])["AUC"].mean().unstack("protocole")
    der = derive_moyenne(d)
    b_rel = auc.xs("relatives")["lomo"].unstack("modele")
    m("resSeqGainMax", f3((b_rel["hgb_seq"] - b_rel["hgb_agg"]).max()))
    gru = b_rel["gru"] - b_rel["rf_agg"]
    m("resGruGainMin", f3(gru.min()))
    m("resGruGainMax", f3(gru.max()))
    d25 = der.xs(0.25, level="cutoff")
    m("resDriftAbsModelsMin", f3(d25.xs("absolues").min()))
    m("resDriftAbsModelsMax", f3(d25.xs("absolues").max()))
    m("resDriftRelHgb", f3(d25[("relatives", "hgb_agg")]))
    m("resDriftRelGru", f3(d25[("relatives", "gru")]))
    c = d[d.cutoff == 0.25].pivot_table(index=["variables", "modele", "module"], columns="protocole",
                                        values="signalement")
    ggg = (c["lomo"] - c["aleatoire"]).xs("GGG", level="module")
    m("resGGGDriftModelsMin", signe(ggg.xs("absolues").min(), 2))
    m("resGGGDriftModelsMax", signe(ggg.xs("absolues").max(), 2))
    a25 = auc.xs(0.25, level="cutoff")
    m("resCostGruRel", f3(a25.loc[("relatives", "gru"), "aleatoire"] - a25.loc[("relatives", "gru"), "lomo"]))

    noms = {"rf_agg": "Forest", "hgb_agg": "Boosting", "hgb_seq": "Boosting + seq.", "gru": "GRU"}
    corps = []
    for v, vl in [("absolues", "Absolute"), ("relatives", "Relative")]:
        for i, mod in enumerate(["rf_agg", "hgb_agg", "hgb_seq", "gru"]):
            r = a25.loc[(v, mod)]
            tete = f"\\multirow{{4}}{{*}}{{{vl}}}" if i == 0 else ""
            corps.append(f"{tete} & {noms[mod]} & {f3(r.aleatoire)} & {f3(r.lomo)} & "
                         f"{signe(r.aleatoire - r.lomo)} & {f3(d25[(v, mod)])} & {signe(ggg[(v, mod)], 2)} \\\\")
        corps.append("\\midrule")
    ecrire("tab_sequence.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Model families and temporal information at the 25\% cut-off (students still enrolled; means over modules). Forest and Boosting: aggregated features; Boosting + seq.: aggregated features plus weekly clicks and active days over the complete weeks before the cut-off and trend features; GRU: recurrent network on the weekly sequence plus all static features. Cost: AUC (A) $-$ AUC (B); Drift: mean absolute flag-rate drift; GGG: drift in module GGG.}}
\label{{tab:sequence}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\toprule
Features & Model & AUC (A) & AUC (B) & Cost & Drift & GGG \\
\midrule
{chr(10).join(corps[:-1])}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def revision():
    """Chiffres et tableaux de la révision (D12–D15) : incertitude, calibration,
    seuil, chevauchement, normalisations, taille de cohorte."""
    # Coûts par module (remplace des valeurs écrites en dur)
    t = pd.read_csv(RES / f"transfer_by_module{META}.csv", index_col=0)
    m("resNCostSmall", str(int((t.cout_transfert_auc < 0.015).sum())))
    m("resAAACost", f3(t.loc["AAA", "cout_transfert_auc"]))
    m("resGGGCost", f3(t.loc["GGG", "cout_transfert_auc"]))

    # 1. IC bootstrap apparié
    b = pd.read_csv(RES / "robustesse_bootstrap_c25.csv").set_index(["quantite", "comparaison"])
    for q, nom in [("AUC (B)", "AucB"), ("Coût", "Cost"), ("|Dérive|", "Drift")]:
        r = b.loc[(q, "relatives − absolues")]
        m(f"resCIDiff{nom}", signe(r.estimation))
        m(f"resCIDiff{nom}CI", f"[{signe(r.ic_bas)}, {signe(r.ic_haut)}]")
        for v, vn in [("absolues", "Abs"), ("relatives", "Rel")]:
            r = b.loc[(q, v)]
            m(f"resCI{nom}{vn}", f"[{f3(r.ic_bas)}, {f3(r.ic_haut)}]")

    # 2. Niveau module
    tm = pd.read_csv(RES / "robustesse_modules_c25.csv", header=[0, 1], index_col=0)
    for q, qn in [("|Dérive|", "Drift"), ("Coût", "Cost")]:
        for v, vn in [("absolues", "Abs"), ("relatives", "Rel")]:
            m(f"resCCC{qn}{vn}", f3(tm.loc["CCC", (q, v)]))
    w = pd.read_csv(RES / "robustesse_wilcoxon_c25.csv").set_index("quantite")
    for q, nom in [("AUC (B)", "Auc"), ("|Dérive|", "Drift")]:
        m(f"resNImproved{nom}", str(int(w.loc[q, "modules_ameliores"])))
        m(f"resWilcoxon{nom}", f"{w.loc[q, 'wilcoxon_p']:.3f}")

    # 3. Graines
    g = pd.read_csv(RES / "robustesse_graines_c25.csv")
    m("resNSeeds", str(pd.read_csv(RES / "seed_variability_c25.csv").graine.nunique()))
    m("resSeedSdAucMin", f3(g.auc_B_et.min()))
    m("resSeedSdAucMax", f3(g.auc_B_et.max()))
    m("resSeedSdDriftMax", f3(g.derive_et.max()))
    gd = pd.read_csv(RES / "robustesse_graines_diff_c25.csv", index_col=0)
    for mod in ["CCC", "EEE", "AAA", "BBB"]:
        m(f"resRelMinusAbs{mod}", signe(gd.loc[mod, "rel_moins_abs_auc_B_mean"]))
        m(f"resRelMinusAbs{mod}Sd", f3(gd.loc[mod, "rel_moins_abs_auc_B_std"]))

    # 4. Calibration : (B) − (A), moyenne des |écarts| sur les modules
    c = pd.read_csv(RES / "robustesse_calibration_c25.csv")
    p = c.pivot_table(index=["variables", "module"], columns="protocole", values=["brier", "citl", "p_moyen", "pente"])
    lignes = []
    for v, vn in [("absolues", "Abs"), ("relatives", "Rel")]:
        ligne = {}
        for q, qn in [("brier", "Brier"), ("citl", "Citl"), ("p_moyen", "Pmean"), ("pente", "Slope")]:
            d = (p[(q, "lomo")] - p[(q, "aleatoire")]).xs(v)
            m(f"resCal{qn}{vn}", f3(d.abs().mean()) if q != "citl" else f2(d.abs().mean()))
            m(f"resCal{qn}GGG{vn}", signe(d["GGG"], 3 if q != "citl" else 2))
            ligne[qn] = (d.abs().mean(), d["GGG"])
        lignes.append((v, ligne))
    corps = []
    for v, ligne in lignes:
        lib = "Absolute" if v == "absolues" else "Relative"
        corps.append(f"{lib} & " + " & ".join(f"{x:.3f} & {signe(y, 3)}" for x, y in
                                               [ligne[k] for k in ["Brier", "Citl", "Pmean", "Slope"]]) + r" \\")
    ecrire("tab_calibration.tex", r"""\begin{table*}[t]
\centering
\caption{Calibration of the transferred model at the 25\% cut-off: difference between protocol (B) and protocol (A) on the same students, as the mean absolute difference over modules and for module GGG. Brier score; calibration-in-the-large (intercept of a logistic recalibration with $\mathrm{logit}(\hat{p})$ as offset); mean predicted risk; calibration slope. With balanced class weights the probabilities are not calibrated even in (A), so the transfer effect is the difference (B) $-$ (A).}
\label{tab:calibration}
\footnotesize
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{Brier} & \multicolumn{2}{c}{Calibration-in-the-large} & \multicolumn{2}{c}{Mean predicted risk} & \multicolumn{2}{c}{Slope} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Features & mean $|\Delta|$ & GGG & mean $|\Delta|$ & GGG & mean $|\Delta|$ & GGG & mean $|\Delta|$ & GGG \\
\midrule
""" + "\n".join(corps) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

    # 5. Seuil
    s = pd.read_csv(RES / "robustesse_seuils_c25.csv")
    num = s[s.seuil != "score moyen"].assign(seuil=lambda x: x.seuil.astype(float))
    for v, vn in [("absolues", "Abs"), ("relatives", "Rel")]:
        x = num[num.variables == v]
        m(f"resThrDrift{vn}Min", f3(x.derive_moy_abs.min()))
        m(f"resThrDrift{vn}Max", f3(x.derive_moy_abs.max()))
        sm = s[(s.variables == v) & (s.seuil == "score moyen")].iloc[0]
        m(f"resScoreShift{vn}", f3(sm.derive_moy_abs))
        m(f"resScoreShiftGGG{vn}", signe(sm.derive_GGG, 2))
    x = num[num.variables == "absolues"]
    m("resThrGGGAbsMin", signe(x.derive_GGG.min(), 2))
    m("resThrGGGAbsMax", signe(x.derive_GGG.max(), 2))

    # 6. Chevauchement : LOMO strict vs standard, contrôle
    o = pd.read_csv(RES / "overlap_check.csv")
    o25 = o[o.cutoff == 0.25]
    m("resOverlapShareMax", pct(o25.part_chevauchement.max(), 1))
    m("resOverlapShareMaxMod", o25.loc[o25.part_chevauchement.idxmax(), "module"])
    m("resOverlapStudents", pct(_part_multi_modules(), 1))
    diff = (o.groupby(["cutoff", "variables"]).auc_B_standard.mean() - o.groupby(["cutoff", "variables"]).auc_B_strict.mean())
    m("resOverlapMeanDiffMax", f3(diff.abs().max()))
    ctl = pd.read_csv(RES / "overlap_control_c25.csv")
    r = ctl.pivot_table(index=["variables", "module", "graine"], columns="condition", values="auc_B")
    eff = pd.DataFrame({"communs": r.communs - r.standard, "hasard": r.hasard - r.standard}).groupby(
        level=["variables", "module"]).mean()
    for mod in ["CCC", "EEE"]:
        m(f"resOverlapEffect{mod}", signe(eff.loc[("absolues", mod), "communs"]))
        m(f"resOverlapRandom{mod}", signe(eff.loc[("absolues", mod), "hasard"]))

    # 7. Normalisations
    nz = pd.read_csv(RES / "normalisations.csv")
    nz["cout"] = nz.auc_A - nz.auc_B
    nz["derive_meme"] = (nz.signal_B - nz.signal_A).abs()
    nz["ecart_risque"] = (nz.signal_B - nz.risque).abs()
    moy = nz[nz.cutoff == 0.25].groupby("variante")[["auc_A", "auc_B", "cout", "derive_meme", "ecart_risque",
                                                    "precision_B", "F1_B"]].mean()
    for v, vn in [("zscore", "Zscore"), ("robuste", "Robust"), ("rang", "Rank")]:
        m(f"resNorm{vn}AucB", f3(moy.loc[v, "auc_B"]))
        m(f"resNorm{vn}Drift", f3(moy.loc[v, "derive_meme"]))
    bmod = nz.pivot_table(index=["cutoff", "module"], columns="variante", values="auc_B")
    ecarts = (bmod[["rang", "zscore", "robuste"]].max(axis=1) - bmod[["rang", "zscore", "robuste"]].min(axis=1))
    moy_c = nz.groupby(["cutoff", "variante"]).auc_B.mean().unstack()
    m("resNormMaxSpread", f3((moy_c[["rang", "zscore", "robuste"]].max(axis=1)
                              - moy_c[["rang", "zscore", "robuste"]].min(axis=1)).max()))

    # 8. Taille de cohorte
    cs = pd.read_csv(RES / "cohort_size.csv")
    cs["abs_ecart"] = cs.ecart_au_complet.abs()
    res = cs.groupby(["taille", "module"])[["AUC", "abs_ecart", "ecart_risque"]].mean().groupby(level="taille").mean()
    for n, nn in [("10", "Ten"), ("20", "Twenty"), ("50", "Fifty"), ("100", "Hundred"), ("complet", "Full")]:
        m(f"resCohortAuc{nn}", f3(res.loc[n, "AUC"]))
        m(f"resCohortFlag{nn}", f3(res.loc[n, "abs_ecart"]))

    noms = {"absolues": "Absolute values", "rang": "Percentile rank", "zscore": "z-score (winsorized 1--99\\%)",
            "robuste": "Robust (median, IQR)"}
    corps = [f"{noms[v]} & {f3(r.auc_A)} & {f3(r.auc_B)} & {signe(r.cout)} & {f3(r.derive_meme)} & "
             f"{f3(r.ecart_risque)} & {f3(r.precision_B)} & {f3(r.F1_B)} \\\\"
             for v, r in moy.reindex(["absolues", "rang", "zscore", "robuste"]).iterrows()]
    taille = "; ".join(f"{n}: {f3(res.loc[n, 'AUC'])}" for n in ["10", "20", "50", "100", "200"])
    ecrire("tab_normalisation.tex", r"""\begin{table*}[t]
\centering
\caption{Which cohort normalization, and how large a cohort? Top: normalization of the eight engagement and early-assessment features within each presentation (25\% cut-off, LOMO strict, means over modules). Drift: mean absolute flag-rate drift (B) $-$ (A) of the same variant; $|$flagged $-$ risk$|$: distance between the share flagged in (B) and the observed at-risk rate. Bottom: AUC (B) of the rank model when ranks are computed in random sub-cohorts of $n$ students of the target module (5 random partitions); with the whole presentation: """ + f3(res.loc["complet", "AUC"]) + r""".}
\label{tab:normalisation}
\footnotesize
\begin{tabular}{@{}lccccccc@{}}
\toprule
Normalization & AUC (A) & AUC (B) & Cost & Drift & $|$flagged $-$ risk$|$ & Precision (B) & F1 (B) \\
\midrule
""" + "\n".join(corps) + r"""
\midrule
\multicolumn{8}{@{}l}{Cohort size $n$ (AUC (B)) --- """ + taille + r"""} \\
\bottomrule
\end{tabular}
\end{table*}
""")


def _part_multi_modules():
    p = pd.read_parquet(RES / "preds" / "preds_rf_active_c25.parquet")
    return (p.groupby("id_student").code_module.nunique() > 1).mean()


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
    sequence()
    revision()
    lignes = ["% Généré par src/08_paper_assets.py — ne pas modifier à la main."]
    lignes += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    ecrire("macros.tex", "\n".join(lignes) + "\n")
    print(f"{len(MACROS)} macros et 10 tableaux écrits dans {OUT}/")


if __name__ == "__main__":
    main()

"""Prépare les archives de l'article pour JLA et pour Overleaf.

Sans option : dossier de soumission anonyme (évaluation en double aveugle).
    JLA demande, pour une soumission LaTeX, le PDF et les sources. Les sources de
    main.tex et supplementary.tex contiennent les noms des auteurs dans les branches
    \\else des blocs \\ifanonymous (version finale) : elles sont retirées, l'archive
    est recompilée dans un dossier isolé et l'absence de toute information
    identifiante est vérifiée, dans les sources comme dans les PDF.
    Sortie : paper/submission/manuscript_anonymous.pdf, supplementary_anonymous.pdf,
             sources_anonymous.zip

--overleaf : projet complet à téléverser sur Overleaf (New Project > Upload Project)
    pour la relecture par les co-auteurs. Les sources sont gardées telles quelles
    (interrupteur \\anonymoustrue / \\anonymousfalse compris), fichiers à la racine.
    Sortie : paper/submission/overleaf_project.zip

Le dossier paper/submission/ n'est pas versionné.
Usage : python paper/make_submission.py [--overleaf]
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FIGURES = PAPER.parent / "figures" / "en"
OUT = PAPER / "submission"

# Chaînes qui ne doivent apparaître ni dans les sources ni dans les PDF anonymes.
IDENTIFIANTS = ["Coly", "Nguer", "Touré", "Tour\\'e", "UN-CHK", "UCAD", "unchk", "ucad.edu",
                "Senegal", "Dakar", "duducoly", "analyses_oualad", "zenodo.24023694", "Doctorat"]

BLOC = re.compile(r"(?<!newif)\\ifanonymous(.*?)\\else(.*?)\\fi", re.S)


def anonymiser(tex: str) -> str:
    """Garde la branche anonyme de chaque bloc \\ifanonymous ... \\else ... \\fi."""
    if r"\anonymoustrue" not in tex:
        sys.exit("ÉCHEC : \\anonymoustrue absent, la version finale ne doit pas partir en évaluation.")
    return BLOC.sub(lambda m: m.group(1), tex)


def rassembler(dest: Path, anonyme: bool) -> list[Path]:
    """Copie dans dest tout ce qu'il faut pour compiler l'article et le supplément."""
    (dest / "generated").mkdir(parents=True)
    (dest / "figures").mkdir()
    for nom in ["main.tex", "supplementary.tex"]:
        tex = (PAPER / nom).read_text(encoding="utf-8")
        if anonyme:
            tex = anonymiser(tex)
        tex = tex.replace(r"\graphicspath{{../figures/en/}}", r"\graphicspath{{figures/}}")
        tex = tex.replace("% The IEEE version of this article is kept in ieee/main.tex.\n", "")
        (dest / nom).write_text(tex, encoding="utf-8")
        for inc in re.findall(r"\\input\{generated/([^}]+)\}", tex):
            shutil.copy(PAPER / "generated" / f"{inc}.tex", dest / "generated" / f"{inc}.tex")
        for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
            shutil.copy(FIGURES / f"{fig}.pdf", dest / "figures" / f"{fig}.pdf")
    for nom in ["JLA_article.cls", "header.png", "references.bib"]:
        shutil.copy(PAPER / nom, dest / nom)
    return sorted(p for p in dest.rglob("*") if p.is_file())


def compiler(dossier: Path):
    """Compile le supplément puis l'article : vérifie que l'archive se suffit à elle-même."""
    for doc in ["supplementary.tex", "main.tex"]:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-pdflatex=pdflatex --disable-installer %O %S", doc],
                           cwd=dossier, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        if r.returncode != 0:
            sys.exit(f"ÉCHEC : compilation de {doc} dans {dossier} (voir le .log)")


def zipper(fichiers: list[Path], racine: Path, cible: Path):
    with zipfile.ZipFile(cible, "w", zipfile.ZIP_DEFLATED) as z:
        for p in fichiers:
            z.write(p, p.relative_to(racine).as_posix())


def soumission():
    if OUT.exists():
        for p in OUT.glob("*anonymous*"):
            p.unlink()
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        fichiers = rassembler(src, anonyme=True)
        compiler(src)

        import pymupdf
        problemes = []
        for p in fichiers:
            if p.suffix in {".tex", ".bib", ".cls"}:
                texte = p.read_text(encoding="utf-8")
                problemes += [f"{p.name}: {i}" for i in IDENTIFIANTS if i in texte]
        for doc in ["main.pdf", "supplementary.pdf"]:
            pdf = pymupdf.open(src / doc)
            texte = " ".join(page.get_text() for page in pdf) + " " + " ".join(str(v) for v in pdf.metadata.values())
            problemes += [f"{doc}: {i}" for i in IDENTIFIANTS if i in texte]
            pdf.close()
        if problemes:
            sys.exit("ÉCHEC : informations identifiantes trouvées :\n  " + "\n  ".join(problemes))

        shutil.copy(src / "main.pdf", OUT / "manuscript_anonymous.pdf")
        shutil.copy(src / "supplementary.pdf", OUT / "supplementary_anonymous.pdf")
        zipper(fichiers, src, OUT / "sources_anonymous.zip")
    print(f"Dossier de soumission prêt : {OUT}")
    for nom in ["manuscript_anonymous.pdf", "supplementary_anonymous.pdf", "sources_anonymous.zip"]:
        print(f"  {nom}  ({(OUT / nom).stat().st_size / 1024:.0f} Ko)")


def overleaf():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        fichiers = rassembler(src, anonyme=False)
        compiler(src)
        zipper(fichiers, src, OUT / "overleaf_project.zip")
    cible = OUT / "overleaf_project.zip"
    print(f"Projet Overleaf prêt : {cible}  ({cible.stat().st_size / 1024:.0f} Ko, {len(fichiers)} fichiers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overleaf", action="store_true", help="projet complet pour Overleaf")
    overleaf() if ap.parse_args().overleaf else soumission()

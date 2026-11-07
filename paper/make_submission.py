"""Prépare le dossier de soumission anonyme pour JLA (évaluation en double aveugle).

JLA demande, pour une soumission LaTeX, le PDF et les sources. Les sources de
main.tex et supplementary.tex contiennent les noms des auteurs dans les branches
\\else des blocs \\ifanonymous (version finale) : ce script les retire, rassemble
les fichiers nécessaires, recompile dans un dossier isolé et vérifie qu'aucune
information identifiante ne reste, dans les sources comme dans les PDF.

Sortie (non versionnée) : paper/submission/
    manuscript_anonymous.pdf, supplementary_anonymous.pdf, sources_anonymous.zip

Usage : python paper/make_submission.py   (après compilation normale de l'article)
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FIGURES = PAPER.parent / "figures" / "en"
OUT = PAPER / "submission"
SRC = OUT / "sources"

# Chaînes qui ne doivent apparaître ni dans les sources ni dans les PDF anonymes.
IDENTIFIANTS = ["Coly", "Nguer", "Touré", "Tour\\'e", "UN-CHK", "UCAD", "unchk", "ucad.edu",
                "Senegal", "Dakar", "duducoly", "analyses_oualad", "zenodo.24023694", "Doctorat"]

BLOC = re.compile(r"(?<!newif)\\ifanonymous(.*?)\\else(.*?)\\fi", re.S)


def anonymiser(tex: str) -> str:
    """Garde la branche anonyme de chaque bloc \\ifanonymous ... \\else ... \\fi."""
    if r"\anonymoustrue" not in tex:
        sys.exit("ÉCHEC : \\anonymoustrue absent, la version finale ne doit pas partir en évaluation.")
    return BLOC.sub(lambda m: m.group(1), tex)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (SRC / "generated").mkdir(parents=True)
    (SRC / "figures").mkdir()

    for nom in ["main.tex", "supplementary.tex"]:
        tex = anonymiser((PAPER / nom).read_text(encoding="utf-8"))
        tex = tex.replace(r"\graphicspath{{../figures/en/}}", r"\graphicspath{{figures/}}")
        tex = tex.replace("% The IEEE version of this article is kept in ieee/main.tex.\n", "")
        (SRC / nom).write_text(tex, encoding="utf-8")
        for inc in re.findall(r"\\input\{generated/([^}]+)\}", tex):
            shutil.copy(PAPER / "generated" / f"{inc}.tex", SRC / "generated" / f"{inc}.tex")
        for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
            shutil.copy(FIGURES / f"{fig}.pdf", SRC / "figures" / f"{fig}.pdf")
    for nom in ["JLA_article.cls", "header.png", "references.bib"]:
        shutil.copy(PAPER / nom, SRC / nom)

    # Compilation dans le dossier isolé : vérifie que l'archive se suffit à elle-même.
    for doc in ["supplementary.tex", "main.tex"]:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-pdflatex=pdflatex --disable-installer %O %S", doc],
                           cwd=SRC, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        if r.returncode != 0:
            sys.exit(f"ÉCHEC : compilation de {doc} dans {SRC} (voir le .log)")

    # Vérification : aucune information identifiante dans les sources ni dans les PDF.
    import pymupdf
    problemes = []
    sources = [p for p in SRC.rglob("*") if p.suffix in {".tex", ".bib", ".cls"}]
    for p in sources:
        texte = p.read_text(encoding="utf-8")
        problemes += [f"{p.name}: {i}" for i in IDENTIFIANTS if i in texte]
    for doc in ["main.pdf", "supplementary.pdf"]:
        pdf = pymupdf.open(SRC / doc)
        texte = " ".join(page.get_text() for page in pdf) + " " + " ".join(str(v) for v in pdf.metadata.values())
        problemes += [f"{doc}: {i}" for i in IDENTIFIANTS if i in texte]
        pdf.close()
    if problemes:
        sys.exit("ÉCHEC : informations identifiantes trouvées :\n  " + "\n  ".join(problemes))

    shutil.copy(SRC / "main.pdf", OUT / "manuscript_anonymous.pdf")
    shutil.copy(SRC / "supplementary.pdf", OUT / "supplementary_anonymous.pdf")
    garder = sources + [SRC / "header.png"] + list((SRC / "figures").glob("*.pdf"))
    with zipfile.ZipFile(OUT / "sources_anonymous.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(garder):
            z.write(p, p.relative_to(SRC).as_posix())
    shutil.rmtree(SRC)
    print(f"Dossier de soumission prêt : {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()

"""Build paper/arxiv.tar.gz: main.tex, main.bbl, generated macros and tables, figures. Run after latexmk.

    .venv/bin/python paper/make_arxiv.py
"""
import glob
import os
import re
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "main.tex")).read()
names = ["main.tex", "main.bbl", "generated/numbers.tex"]
names += sorted(f"generated/{n}.tex" for n in set(re.findall(r"\\input\{generated/(tab_\w+)\.tex\}", tex)))
names += sorted(f"figures/{n}" for n in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
with tarfile.open(os.path.join(HERE, "arxiv.tar.gz"), "w:gz") as t:
    for n in names:
        t.add(os.path.join(HERE, n), arcname=n)
print(len(names), "files;", len(glob.glob(os.path.join(HERE, "figures", "*.pdf"))), "figures on disk")

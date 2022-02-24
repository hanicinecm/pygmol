import os
import sys
import re
from pathlib import Path

file_dir = Path(__file__).absolute().parent
docs_dir = file_dir.parent
project_dir = docs_dir.parent

sys.path.append(str(project_dir))

doc_name = "pygmol_doc"

# update the package version:
with open(project_dir.joinpath("setup.py")) as fp:
    setup_text = fp.read()
# for some reason, I cannot import from setup.py :(
version = re.search(r'version\s?=\s?"(\d{1,2}\.\d{1,2}\.\d{1,2})"', setup_text).group(1)
with open(file_dir.joinpath(f"{doc_name}.tex")) as fp:
    text = fp.read()
updated = re.sub(r"\d{1,2}\.\d{1,2}\.\d{1,2}", version, text)
with open(file_dir.joinpath(f"{doc_name}.tex"), "w") as fp:
    fp.write(updated)

# run the latex commands:
os.chdir(file_dir)
commands = [
    f"pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex",
    f"bibtex {doc_name}.aux",
    f"pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex",
    f"pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex",
]
for cmd in commands:
    os.system(cmd)

# remove the byproducts:
to_keep = set([f"{doc_name}.{ext}" for ext in ["tex", "pdf", "bib"]])
to_keep.add(f"{doc_name}_content.tex")
for file_path in file_dir.glob(f"{doc_name}*.*"):
    if file_path.name not in to_keep:
        file_path.unlink()

# move the final pdf one folder up and rename:
file_dir.joinpath(f"{doc_name}.pdf").rename(docs_dir.joinpath("math.pdf"))

# Configuration file for the Sphinx documentation builder.

# -- Project information
project = "3DMolMS"
copyright = "2023, Yuhui Hong"
author = "Yuhui Hong"

# Single source of truth: read the package version from molnetpack/_version.py.
# File read (not import) so the docs build without torch installed.
import re
from pathlib import Path
version = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (Path(__file__).resolve().parents[2] / "molnetpack" / "_version.py").read_text(),
).group(1)
release = version

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

# For autodoc to work with modules that are not installed, we need to mock them before import molnetpack.
autodoc_mock_imports = [
    # External packages
    "requests",
    "numpy",
    "pandas",
    "yaml",
    "pyteomics",
    "zipfile",
    "rdkit",
    "PIL",
    "matplotlib",
    "torch",
    "torch.nn",
    "torch.optim",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

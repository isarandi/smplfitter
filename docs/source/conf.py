# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SMPLFitter'
copyright = '2024, István Sárándi'
author = 'István Sárándi'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['abbrev_long.bib', 'references.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "SMPLFitter"
html_css_files = ['custom.css']

html_theme_options = {
    "repository_url": "https://github.com/isarandi/smplfit",  # optional, adjust to your repo if applicable
    "use_repository_button": True,                     # optional, if you want GitHub link
    "use_download_button": False,                      # optional, remove download button
    "use_fullscreen_button": True,                     # optional, add fullscreen button
}
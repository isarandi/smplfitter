import datetime
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../src'))

from conf_spec import project, project_slug, release

release = release

# -- Project information -----------------------------------------------------
linkcode_url = f'https://github.com/isarandi/{project_slug}'

author = 'István Sárándi'
copyright = f'{datetime.datetime.now().year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

add_module_names = False
python_use_unqualified_type_names = True
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc.typehints',
    'sphinxcontrib.bibtex',
    'autoapi.extension',
    'sphinx.ext.viewcode',
    "sphinx_markdown_builder",
    'sphinx.ext.inheritance_diagram',
]
bibtex_bibfiles = ['abbrev_long.bib', 'references.bib']
bibtex_footbibliography_header = ".. rubric:: References"
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/main/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

github_username = 'isarandi'
github_repository = project_slug
autodoc_show_sourcelink = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
python_display_short_literal_types = True

html_title = project
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_toc_level": 3,
}
html_static_path = ['_static']
html_css_files = ['styles/my_theme.css']
toc_object_entries_show_parents = "hide"

autoapi_root = 'api'
autoapi_member_order = 'bysource'
autodoc_typehints = 'description'
autoapi_own_page_level = 'attribute'
autoapi_type = 'python'
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'undoc-members': False,
    'exclude-members': '__init__, __weakref__, __repr__, __str__'
}
autoapi_options = ['members', 'show-inheritance', 'special-members', 'show-module-summary']
autoapi_add_toctree_entry = True
autoapi_dirs = ['../src']
autoapi_template_dir = '_templates/autoapi'

autodoc_member_order = 'bysource'
autoclass_content = 'class'

autosummary_generate = True
autosummary_imported_members = False


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Skip members (functions, classes, modules) without docstrings.
    """
    # Check if the object has a __doc__ attribute
    if not getattr(obj, 'docstring', None):
        return True  # Skip if there's no docstring
    elif what in ('class', 'function'):
        # Check if the module of the class has a docstring
        module_name = '.'.join(name.split('.')[:-1])
        try:
            module = importlib.import_module(module_name)
            return not getattr(module, '__doc__', None)
        except ModuleNotFoundError:
            pass


def setup(app):
    app.connect('autoapi-skip-member', autodoc_skip_member)
    app.connect('autodoc-skip-member', autodoc_skip_member)

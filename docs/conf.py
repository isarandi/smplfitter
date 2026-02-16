"""Sphinx configuration for smplfitter documentation."""

import types
import contextlib
import importlib
import inspect
import os
import re
import sys
from enum import Enum

import setuptools_scm
import toml

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Read project info from pyproject.toml
pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'))

with open(pyproject_path) as f:
    data = toml.load(f)

project_info = data['project']
project_slug = project_info['name'].replace(' ', '-').lower()
tool_urls = project_info.get('urls', {})

repo_url = tool_urls.get('Repository', '')
author_url = tool_urls.get('Author', '')

# Extract GitHub username from repo URL
github_match = re.match(r'https://github\.com/([^/]+)/?', repo_url)
github_username = github_match[1] if github_match else ''

project = project_info['name']
release = setuptools_scm.get_version('..')
version = '.'.join(release.split('.')[:2])
main_module_name = project_slug.replace('-', '_')
repo_name = project_slug
module = importlib.import_module(main_module_name)
globals()[main_module_name] = module

# -- Project information -----------------------------------------------------
linkcode_url = repo_url

author = project_info['authors'][0]['name']
copyright = '2024-%Y'

# -- General configuration ---------------------------------------------------

add_module_names = False
python_use_unqualified_type_names = True

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.autodoc.typehints',
    'sphinxcontrib.bibtex',
    'autoapi.extension',
    'sphinx.ext.inheritance_diagram',
    'sphinx_codeautolink',
]

bibtex_bibfiles = ['abbrev_long.bib', 'references.bib']
bibtex_footbibliography_header = '.. rubric:: References'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/main/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

github_username = github_username
github_repository = repo_name
autodoc_show_sourcelink = False
html_show_sourcelink = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------

html_title = project
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'show_toc_level': 3,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': repo_url,
            'icon': 'fa-brands fa-square-github',
            'type': 'fontawesome',
        }
    ],
}
html_static_path = ['_static']
html_css_files = ['styles/my_theme.css']

html_context = {
    'author_url': author_url,
    'author': author,
}

# -- AutoAPI configuration ---------------------------------------------------

autoapi_root = 'api'
autoapi_member_order = 'bysource'
autodoc_typehints = 'description'
autoapi_own_page_level = 'attribute'
autoapi_type = 'python'

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'undoc-members': False,
    'exclude-members': '__init__, __weakref__, __repr__, __str__',
}

autoapi_options = ['members', 'show-inheritance', 'special-members', 'show-module-summary']
autoapi_add_toctree_entry = False
autoapi_dirs = ['../src']
autoapi_template_dir = '_templates/autoapi'

autodoc_member_order = 'bysource'
autoclass_content = 'class'

autosummary_generate = True
autosummary_imported_members = False

toc_object_entries_show_parents = 'hide'
python_display_short_literal_types = True


# -- Skip undocumented members -----------------------------------------------


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip members without docstrings or from undocumented modules."""
    if not getattr(obj, 'docstring', None):
        return True
    elif what in ('class', 'function', 'attribute'):
        module_name = '.'.join(name.split('.')[:-1])
        try:
            mod = importlib.import_module(module_name)
            if not getattr(mod, '__doc__', None):
                return True  # Module has no docstring, skip its members
        except ModuleNotFoundError:
            # Import failed (e.g. attribute path like Class.attr, or missing dep).
            # Defer to AutoAPI default.
            return None
        # For private names, defer to AutoAPI default (which skips them).
        # For public names, force-include (overrides the imported-members default).
        short_name = name.split('.')[-1]
        if short_name.startswith('_') and not short_name.startswith('__'):
            return None
        return False
    return skip


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None

    fullname = info['fullname']
    try:
        file, start, end = get_line_numbers(eval(fullname))
    except AttributeError:
        # Instance attribute (dataclass field or self.x = ... in __init__)
        parts = fullname.rsplit('.', 1)
        if len(parts) != 2:
            return None
        try:
            file, start, end = get_attr_line_numbers(eval(parts[0]), parts[1])
        except Exception:
            return None
    except Exception as e:
        print(f'linkcode_resolve failed: {info} — {e}')
        return None

    relpath = os.path.relpath(file, os.path.dirname(module.__file__))
    return f'{repo_url}/blob/v{release}/src/{main_module_name}/{relpath}#L{start}-L{end}'


def get_line_numbers(obj):
    if isinstance(obj, property):
        obj = obj.fget

    if isinstance(obj, Enum):
        return get_enum_member_line_numbers(obj)

    if inspect.ismemberdescriptor(obj):
        return get_member_line_numbers(obj)

    with module_restored(obj):
        lines = inspect.getsourcelines(obj)
        file = inspect.getsourcefile(obj)

    start, end = lines[1], lines[1] + len(lines[0]) - 1
    return file, start, end


def get_enum_member_line_numbers(obj):
    class_ = obj.__class__
    with module_restored(class_):
        source_lines, start_line = inspect.getsourcelines(class_)

        for i, line in enumerate(source_lines):
            if f'{obj.name} =' in line:
                return inspect.getsourcefile(class_), start_line + i, start_line + i
        else:
            raise ValueError(f'Enum member {obj.name} not found in {class_}')


def get_attr_line_numbers(class_, attr_name):
    with module_restored(class_):
        source_lines, start_line = inspect.getsourcelines(class_)
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if (
                stripped.startswith(f'{attr_name}:')
                or stripped.startswith(f'{attr_name} :')
                or f'self.{attr_name} =' in stripped
                or f'self.{attr_name}=' in stripped
            ):
                return inspect.getsourcefile(class_), start_line + i, start_line + i
        else:
            raise ValueError(f'Attribute {attr_name} not found in {class_}')


def get_member_line_numbers(obj: types.MemberDescriptorType):
    class_ = obj.__objclass__
    with module_restored(class_):
        source_lines, start_line = inspect.getsourcelines(class_)

        for i, line in enumerate(source_lines):
            if f'{obj.__name__} = ' in line:
                return inspect.getsourcefile(class_), start_line + i, start_line + i
        else:
            raise ValueError(f'Member {obj.__name__} not found in {class_}')


@contextlib.contextmanager
def module_restored(obj):
    if not hasattr(obj, '_module_original_'):
        yield
    else:
        fake_module = obj.__module__
        obj.__module__ = obj._module_original_
        yield
        obj.__module__ = fake_module


def setup(app):
    """Sphinx setup hook."""
    app.connect('autoapi-skip-member', autodoc_skip_member)
    app.connect('autodoc-skip-member', autodoc_skip_member)

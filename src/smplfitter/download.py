"""Download SMPL-family body model files from official sources.

Usage::

    python -m smplfitter.download [--models smpl smplx ...] [target_directory]

or from Python::

    import smplfitter.download
    smplfitter.download.download(['smpl'])

Without ``--models``, all models are downloaded. Each model needs a registration at the
listed sites beforehand (use the same email and password for all of them):

- ``smpl``: https://smpl.is.tue.mpg.de/ and https://agora.is.tue.mpg.de/ (kid template)
- ``smplx``: https://smpl-x.is.tue.mpg.de/ and https://agora.is.tue.mpg.de/ (kid template)
- ``smplh``: https://mano.is.tue.mpg.de/ and https://agora.is.tue.mpg.de/ (kid template)
- ``mano``: https://mano.is.tue.mpg.de/
- ``correspondences`` (SMPL <-> SMPL-X): https://smpl-x.is.tue.mpg.de/

Without an AGORA account, use ``--no-kid-template``: an all-zero placeholder then stands
in for the kid template, and the body models work with the kid factor disabled
(``enable_kid=False``; enabling it raises an error). A later download without the option
replaces the placeholder.

Models already present are skipped, and the login is asked for only if something is
missing. The email and password can also come from the ``SMPLFITTER_EMAIL`` and
``SMPLFITTER_PASSWORD`` environment variables.
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

MODELS = ('smpl', 'smplx', 'smplh', 'mano', 'correspondences')

_DOMAIN_REGISTRATION_URLS = {
    'smpl': 'https://smpl.is.tue.mpg.de/',
    'smplx': 'https://smpl-x.is.tue.mpg.de/',
    'mano': 'https://mano.is.tue.mpg.de/',
    'agora': 'https://agora.is.tue.mpg.de/',
}


def main():
    parser = argparse.ArgumentParser(
        prog='python -m smplfitter.download',
        description=__doc__.split('Usage::')[0].strip(),
        epilog='Registrations needed (same email and password at all sites):\n'
        + __doc__.split('beforehand (use the same email and password for all of them):')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'target_directory',
        nargs='?',
        help='body_models directory (default: asked, or from SMPLFITTER_BODY_MODELS / DATA_ROOT)',
    )
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument(
        '--no-kid-template',
        dest='kid_template',
        action='store_false',
        help='write an all-zero placeholder instead of downloading the kid template (no '
        'AGORA account needed; kid fitting then unavailable)',
    )
    args = parser.parse_args()

    if args.target_directory:
        target = Path(args.target_directory)
    else:
        target = _resolve_body_models_dir_interactively()
    print(f'Body models will be saved to: {target}')
    try:
        download(args.models, target, kid_template=args.kid_template)
    except PermissionError as e:
        print(f'\n{e}', file=sys.stderr)
        sys.exit(1)
    print()
    print(f'Body models are in {target}')


def download(models=MODELS, body_models_dir=None, email=None, password=None, kid_template=True):
    """Download body model files, skipping the models that are already present.

    Args:
        models: names from :data:`MODELS`. ``smpl``, ``smplx`` and ``smplh`` include the
            kid template, which the body model loader requires for them.
        body_models_dir: the directory that contains (or will contain) ``smpl/``,
            ``smplx/``, etc. Default: :func:`smplfitter.common.body_models_dir`, where the
            loader looks.
        email: account email for the download sites. Default: ``$SMPLFITTER_EMAIL``, else
            asked on the terminal (only if something needs downloading).
        password: account password. Default: ``$SMPLFITTER_PASSWORD``, else asked.
        kid_template: if False, write an all-zero placeholder instead of downloading the kid
            template, so no AGORA account is needed; the body models then work only with
            ``enable_kid=False`` (enabling it raises an error). If True,
            an existing placeholder is replaced by the real file.

    Returns:
        The body_models directory, as a :class:`pathlib.Path`.

    Raises:
        ValueError: for an unknown model name.
        PermissionError: if a site rejects the login (wrong password, or no registration
            at that site).
    """
    if isinstance(models, str):
        models = [models]
    if unknown := sorted(set(models) - set(MODELS)):
        raise ValueError(f'Unknown model(s) {unknown}; choose from {list(MODELS)}')
    if body_models_dir is None:
        from smplfitter.common import body_models_dir as default_dir

        body_models_dir = default_dir()
    body_models_dir = Path(body_models_dir)

    missing = [
        m for m in MODELS if m in models and not _is_present(m, body_models_dir, kid_template)
    ]
    if not missing:
        return body_models_dir
    if not kid_template:
        for model in missing:
            _write_kid_template_placeholder(body_models_dir, model)
        missing = [m for m in missing if not _is_present(m, body_models_dir, kid_template)]
        _link_kid_templates(body_models_dir)
        if not missing:
            return body_models_dir

    email = email or os.getenv('SMPLFITTER_EMAIL')
    password = password or os.getenv('SMPLFITTER_PASSWORD')
    if not (email and password):
        print(f'Downloading: {", ".join(missing)}. This needs an account (same email and')
        print('password) registered at each of these sites:')
        sites = {d for m in missing for d in _SITES[m] if kid_template or d != 'agora'}
        for domain in sorted(sites):
            print(f'  - {_DOMAIN_REGISTRATION_URLS[domain]}')
        email = email or input('Email: ')
        password = password or getpass.getpass('Password: ')

    opener = _make_opener()
    username_enc = urllib.parse.quote(email, safe='')
    password_enc = urllib.parse.quote(password, safe='')
    auth_data = f'username={username_enc}&password={password_enc}'.encode()

    for model in missing:
        _DOWNLOADERS[model](opener, auth_data, body_models_dir)
        if kid_template and model in _KID_TEMPLATE_SOURCE:
            _download_kid_template(opener, auth_data, body_models_dir, model)
    _link_kid_templates(body_models_dir)
    return body_models_dir


# Sites (keys of _DOMAIN_REGISTRATION_URLS) that each model's files come from
_SITES = {
    'smpl': ('smpl', 'agora'),
    'smplx': ('smplx', 'agora'),
    'smplh': ('mano', 'agora'),
    'mano': ('mano',),
    'correspondences': ('smplx',),
}

# The kid template each model uses: SMPL+H shares SMPL's
_KID_TEMPLATE_SOURCE = {'smpl': 'smpl', 'smplx': 'smplx', 'smplh': 'smpl'}
# Shape of the kid templates (one point per vertex of the model they belong to)
_KID_TEMPLATE_SHAPE = {'smpl': (6890, 3), 'smplx': (10475, 3)}


def _is_present(model, body_models_dir, kid_template=True):
    """Whether the model's files exist; with ``kid_template``, a placeholder does not count."""
    files = {
        'smpl': ['smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'],
        'smplx': ['smplx/SMPLX_NEUTRAL.npz'],
        'smplh': ['smplh/SMPLH_female.pkl'],
        'mano': ['mano/MANO_RIGHT.pkl'],
        'correspondences': [
            'smpl2smplx_deftrafo_setup.pkl',
            'smplx2smpl_deftrafo_setup.pkl',
        ],
    }[model]
    if not all((body_models_dir / f).exists() for f in files):
        return False
    if model not in _KID_TEMPLATE_SOURCE:
        return True
    kid_path = body_models_dir / _KID_TEMPLATE_SOURCE[model] / 'kid_template.npy'
    if not kid_path.exists():
        return False
    return not kid_template or not _is_kid_template_placeholder(kid_path)


def _write_kid_template_placeholder(body_models_dir, model):
    """Write the all-zero kid template for ``model``, unless a kid template file exists."""
    if model not in _KID_TEMPLATE_SOURCE:
        return
    import numpy as np

    source = _KID_TEMPLATE_SOURCE[model]
    path = body_models_dir / source / 'kid_template.npy'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros(_KID_TEMPLATE_SHAPE[source], np.float64))
        print(f'[kid template] Wrote an all-zero placeholder to {path} (kid factor disabled).')


def _is_kid_template_placeholder(path):
    import numpy as np

    return not np.load(path).any()


def _resolve_body_models_dir_interactively():
    """Like smplfitter.common.body_models_dir, but confirm the platform default first."""
    body_models_dir = os.getenv('SMPLFITTER_BODY_MODELS')
    if body_models_dir:
        return Path(body_models_dir)

    data_root = os.getenv('DATA_ROOT')
    if data_root:
        return Path(data_root) / 'body_models'

    import platformdirs

    default = Path(platformdirs.user_data_dir('smplfitter')) / 'body_models'
    print('No SMPLFITTER_BODY_MODELS or DATA_ROOT environment variable set.')
    print(f'Default location: {default}')
    answer = input('Use this location? [Y/n] ').strip().lower()
    if answer in ('', 'y', 'yes'):
        return default

    custom = input('Enter path for body_models directory: ').strip()
    return Path(custom)


def _make_opener():
    """Create an HTTPS opener with cookie jar for MPI server authentication."""
    cj = http.cookiejar.CookieJar()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(cj),
    )


def _download_mpi(opener, auth_data, domain, filename, peek_only=False):
    """Download a file from the MPI server with POST authentication."""
    url = (
        f'https://download.is.tue.mpg.de/download.php'
        f'?domain={domain}&resume=1&sfile={urllib.parse.quote(filename)}'
    )
    req = urllib.request.Request(url, data=auth_data, method='POST')
    try:
        response = opener.open(req)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            reg_url = _DOMAIN_REGISTRATION_URLS.get(domain, '???')
            raise PermissionError(
                f'Login rejected for the "{domain}" downloads: wrong email or password, or '
                f'no registration at {reg_url}'
            ) from None
        raise
    if peek_only:
        response.read(1024)
        response.close()
        return None
    return response


def _download_mpi_to_file(opener, auth_data, domain, filename, dest_path):
    """Download a file from the MPI server and save to disk."""
    print(f'  Downloading {filename}...')
    response = _download_mpi(opener, auth_data, domain, filename)
    total = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    with open(dest_path, 'wb') as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f'\r  Downloading {filename}... {pct}%', end='', flush=True)
    print()
    return dest_path


def _download_smpl(opener, auth_data, body_models_dir):
    """Download and extract SMPL model files."""
    smpl_dir = body_models_dir / 'smpl'
    smpl_dir.mkdir(parents=True, exist_ok=True)

    target = smpl_dir / 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
    if target.exists():
        print('[smpl] Already downloaded, skipping.')
        return

    print('[smpl] Downloading SMPL model...')
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'smpl.zip'
        _download_mpi_to_file(opener, auth_data, 'smpl', 'SMPL_python_v.1.1.0.zip', zip_path)
        print('  Extracting...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('basicmodel_') and basename.endswith('.pkl'):
                    _extract_zip_member(zf, member, smpl_dir / basename)

    # Create convenience symlinks
    _symlink(smpl_dir / 'SMPL_MALE.pkl', 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl')
    _symlink(smpl_dir / 'SMPL_FEMALE.pkl', 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl')
    _symlink(smpl_dir / 'SMPL_NEUTRAL.pkl', 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')

    # Download J_regressor files from SPIN
    _download_spin_regressors(smpl_dir)

    print('[smpl] Done.')


def _download_smplx(opener, auth_data, body_models_dir):
    """Download and extract SMPL-X model files."""
    smplx_dir = body_models_dir / 'smplx'
    smplx_dir.mkdir(parents=True, exist_ok=True)
    smplxlh_dir = body_models_dir / 'smplxlh'
    smplxlh_dir.mkdir(parents=True, exist_ok=True)

    target = smplx_dir / 'SMPLX_NEUTRAL.npz'
    if target.exists():
        print('[smplx] Already downloaded, skipping.')
        return

    print('[smplx] Downloading SMPL-X models...')
    with tempfile.TemporaryDirectory() as tmp:
        # Main SMPL-X models
        zip_path = Path(tmp) / 'smplx.zip'
        _download_mpi_to_file(opener, auth_data, 'smplx', 'models_smplx_v1_1.zip', zip_path)
        print('  Extracting SMPL-X models...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('SMPLX_') and basename.endswith('.npz'):
                    _extract_zip_member(zf, member, smplx_dir / basename)

        # Locked-head variant
        zip_path = Path(tmp) / 'smplx_lh.zip'
        _download_mpi_to_file(
            opener, auth_data, 'smplx', 'smplx_lockedhead_20230207.zip', zip_path
        )
        print('  Extracting locked-head models...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('SMPLX_') and basename.endswith('.npz'):
                    _extract_zip_member(zf, member, smplxlh_dir / basename)

        # Flip correspondences
        zip_path = Path(tmp) / 'flip.zip'
        _download_mpi_to_file(
            opener, auth_data, 'smplx', 'smplx_flip_correspondences.zip', zip_path
        )
        print('  Extracting flip correspondences...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.endswith('.npz'):
                    _extract_zip_member(zf, member, smplx_dir / basename)

        # MANO/FLAME vertex correspondences
        zip_path = Path(tmp) / 'smplx_mano_flame_correspondences.zip'
        _download_mpi_to_file(
            opener, auth_data, 'smplx', 'smplx_mano_flame_correspondences.zip', zip_path
        )
        print('  Extracting MANO/FLAME correspondences...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.endswith(('.pkl', '.npy')):
                    _extract_zip_member(zf, member, smplx_dir / basename)

    # Download SMPLX_to_J14 from HuggingFace
    j14_path = smplx_dir / 'SMPLX_to_J14.pkl'
    if not j14_path.exists():
        print('  Downloading SMPLX_to_J14.pkl from HuggingFace...')
        url = 'https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_to_J14.pkl'
        urllib.request.urlretrieve(url, j14_path)

    print('[smplx] Done.')


def _download_smplh(opener, auth_data, body_models_dir):
    """Download and extract SMPL+H model files."""
    smplh_dir = body_models_dir / 'smplh'
    smplh_dir.mkdir(parents=True, exist_ok=True)
    smplh16_dir = body_models_dir / 'smplh16'
    smplh16_dir.mkdir(parents=True, exist_ok=True)

    target = smplh_dir / 'SMPLH_female.pkl'
    if target.exists():
        print('[smplh] Already downloaded, skipping.')
        return

    print('[smplh] Downloading SMPL+H models...')
    with tempfile.TemporaryDirectory() as tmp:
        # MANO package contains SMPLH models
        zip_path = Path(tmp) / 'mano.zip'
        _download_mpi_to_file(opener, auth_data, 'mano', 'mano_v1_2.zip', zip_path)
        print('  Extracting SMPL+H models...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('SMPLH_') and basename.endswith('.pkl'):
                    _extract_zip_member(zf, member, smplh_dir / basename)

        # SMPL+H 16 joints
        tar_path = Path(tmp) / 'smplh.tar.xz'
        _download_mpi_to_file(opener, auth_data, 'mano', 'smplh.tar.xz', tar_path)
        print('  Extracting SMPL+H16 models...')
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                # Expected structure: smplh/{male,female,neutral}/model.npz
                parts = Path(member.name).parts
                if len(parts) >= 2 and parts[-1] == 'model.npz':
                    gender = parts[-2]
                    if gender in ('male', 'female', 'neutral'):
                        dest = smplh16_dir / gender / 'model.npz'
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        _extract_tar_member(tf, member, dest)

    # Create convenience symlinks (some tools expect uppercase gender)
    _symlink(smplh_dir / 'SMPLH_FEMALE.pkl', 'SMPLH_female.pkl')
    _symlink(smplh_dir / 'SMPLH_MALE.pkl', 'SMPLH_male.pkl')

    print('[smplh] Done.')


def _download_mano(opener, auth_data, body_models_dir):
    """Download and extract MANO hand model files."""
    mano_dir = body_models_dir / 'mano'
    mano_dir.mkdir(parents=True, exist_ok=True)

    target = mano_dir / 'MANO_RIGHT.pkl'
    if target.exists():
        print('[mano] Already downloaded, skipping.')
        return

    print('[mano] Downloading MANO models...')
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'mano.zip'
        _download_mpi_to_file(opener, auth_data, 'mano', 'mano_v1_2.zip', zip_path)
        print('  Extracting...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('MANO_') and basename.endswith('.pkl'):
                    _extract_zip_member(zf, member, mano_dir / basename)

    print('[mano] Done.')


def _download_kid_template(opener, auth_data, body_models_dir, model):
    """Download the kid template (from AGORA) that ``model`` uses, if missing."""
    source = _KID_TEMPLATE_SOURCE[model]
    target = body_models_dir / source / 'kid_template.npy'
    if target.exists() and not _is_kid_template_placeholder(target):
        return
    print(f'[kid template] Downloading the {source} kid template from AGORA...')
    target.parent.mkdir(parents=True, exist_ok=True)
    response = _download_mpi(opener, auth_data, 'agora', f'{source}_kid_template.npy')
    data = response.read()
    target.unlink(missing_ok=True)  # may be a symlink to a placeholder
    target.write_bytes(data)
    print(f'  Saved {target}')


def _link_kid_templates(body_models_dir):
    """Link the kid templates into the variant folders (smplh, smplh16, smplxlh)."""
    for subdir, source in (('smplh', 'smpl'), ('smplh16', 'smpl'), ('smplxlh', 'smplx')):
        source_file = body_models_dir / source / 'kid_template.npy'
        target = body_models_dir / subdir / 'kid_template.npy'
        if source_file.exists() and target.parent.exists() and not target.exists():
            _symlink(target, os.path.relpath(source_file, target.parent))


def _download_correspondences(opener, auth_data, body_models_dir):
    """Download SMPL-to-SMPLX correspondence files."""
    smpl2smplx = body_models_dir / 'smpl2smplx_deftrafo_setup.pkl'
    smplx2smpl = body_models_dir / 'smplx2smpl_deftrafo_setup.pkl'

    if smpl2smplx.exists() and smplx2smpl.exists():
        print('[correspondences] Already downloaded, skipping.')
        return

    print('[correspondences] Downloading SMPL↔SMPL-X correspondence files...')
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'correspondences.zip'
        _download_mpi_to_file(
            opener,
            auth_data,
            'smplx',
            'model_transfer.zip',
            zip_path,
        )
        print('  Extracting...')
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if 'deftrafo_setup' in basename and basename.endswith('.pkl'):
                    _extract_zip_member(zf, member, body_models_dir / basename)

    print('[correspondences] Done.')


def _download_spin_regressors(smpl_dir):
    """Download J_regressor files from the SPIN project."""
    j_extra = smpl_dir / 'J_regressor_extra.npy'
    j_h36m = smpl_dir / 'J_regressor_h36m.npy'

    if j_extra.exists() and j_h36m.exists():
        return

    print('  Downloading SPIN J_regressor files...')
    url = 'http://visiondata.cis.upenn.edu/spin/data.tar.gz'
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / 'data.tar.gz'
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                basename = os.path.basename(member.name)
                if basename in ('J_regressor_extra.npy', 'J_regressor_h36m.npy'):
                    _extract_tar_member(tf, member, smpl_dir / basename)


_DOWNLOADERS = {
    'smpl': _download_smpl,
    'smplx': _download_smplx,
    'smplh': _download_smplh,
    'mano': _download_mano,
    'correspondences': _download_correspondences,
}


# --- Helpers ---


def _extract_zip_member(zf, member, dest):
    """Extract a single zip member to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, 'wb') as dst:
        shutil.copyfileobj(src, dst)


def _extract_tar_member(tf, member, dest):
    """Extract a single tar member to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = tf.extractfile(member)
    if src is None:
        return
    with open(dest, 'wb') as dst:
        shutil.copyfileobj(src, dst)


def _symlink(link_path, target):
    """Create a symlink, skipping if it already exists."""
    if not link_path.exists():
        link_path.symlink_to(target)


if __name__ == '__main__':
    main()

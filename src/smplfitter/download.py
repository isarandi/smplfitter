"""Download SMPL-family body model files from official sources.

Usage::

    python -m smplfitter.download [target_directory]

Requires registration at https://smpl.is.tue.mpg.de/ beforehand.
"""

from __future__ import annotations

import getpass
import http.cookiejar
import io
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


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print(__doc__.strip())
        sys.exit(0)

    if len(sys.argv) > 1:
        body_models_dir = Path(sys.argv[1])
    else:
        body_models_dir = _resolve_body_models_dir()

    print(f'Body models will be saved to: {body_models_dir}')
    print()
    print('You need an account at https://smpl.is.tue.mpg.de/')
    print('(Register there first if you have not already.)')
    print()
    email = input('Email: ')
    password = getpass.getpass('Password: ')

    opener = _make_opener()
    username_enc = urllib.parse.quote(email, safe='')
    password_enc = urllib.parse.quote(password, safe='')
    auth_data = f'username={username_enc}&password={password_enc}'.encode()

    # Verify credentials with a small download before doing the big ones
    print()
    print('Verifying credentials...')
    try:
        _download_mpi(opener, auth_data, 'smpl', 'SMPL_python_v.1.1.0.zip', peek_only=True)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print('Authentication failed. Check your email and password.', file=sys.stderr)
            sys.exit(1)
        raise
    print('OK')
    print()

    _download_smpl(opener, auth_data, body_models_dir)
    _download_smplx(opener, auth_data, body_models_dir)
    _download_smplh(opener, auth_data, body_models_dir)
    _download_mano(opener, auth_data, body_models_dir)
    _download_kid_templates(opener, auth_data, body_models_dir)
    _download_correspondences(opener, auth_data, body_models_dir)

    print()
    print('All downloads complete!')
    print(f'Body models saved to: {body_models_dir}')


def _resolve_body_models_dir():
    """Resolve the body models directory from environment variables or default."""
    body_models_dir = os.getenv('SMPLFITTER_BODY_MODELS')
    if body_models_dir:
        return Path(body_models_dir)

    data_root = os.getenv('DATA_ROOT')
    if data_root:
        return Path(data_root) / 'body_models'

    default = Path.home() / '.local' / 'share' / 'smplfitter' / 'body_models'
    print(f'No SMPLFITTER_BODY_MODELS or DATA_ROOT environment variable set.')
    print(f'Default location: {default}')
    answer = input(f'Use this location? [Y/n] ').strip().lower()
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
    response = opener.open(req)
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
        _download_mpi_to_file(
            opener, auth_data, 'smplx', 'models_smplx_v1_1.zip', zip_path
        )
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

    target = smplh_dir / 'SMPLH_FEMALE.pkl'
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


def _download_kid_templates(opener, auth_data, body_models_dir):
    """Download kid templates from AGORA dataset."""
    smpl_kid = body_models_dir / 'smpl' / 'kid_template.npy'
    smplx_kid = body_models_dir / 'smplx' / 'kid_template.npy'

    if smpl_kid.exists() and smplx_kid.exists():
        print('[kid templates] Already downloaded, skipping.')
        return

    print('[kid templates] Downloading from AGORA...')

    if not smpl_kid.exists():
        response = _download_mpi(opener, auth_data, 'agora', 'smpl_kid_template.npy')
        smpl_kid.write_bytes(response.read())
        print(f'  Saved {smpl_kid}')

    if not smplx_kid.exists():
        response = _download_mpi(opener, auth_data, 'agora', 'smplx_kid_template.npy')
        smplx_kid.write_bytes(response.read())
        print(f'  Saved {smplx_kid}')

    # Symlink kid templates for smplh, smplh16, smplxlh
    for subdir in ('smplh', 'smplh16'):
        target = body_models_dir / subdir / 'kid_template.npy'
        if not target.exists() and (body_models_dir / subdir).exists():
            _symlink(target, os.path.relpath(smpl_kid, target.parent))

    smplxlh_dir = body_models_dir / 'smplxlh'
    if smplxlh_dir.exists():
        target = smplxlh_dir / 'kid_template.npy'
        if not target.exists():
            _symlink(target, os.path.relpath(smplx_kid, target.parent))

    print('[kid templates] Done.')


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
            opener, auth_data, 'smplx',
            'smplx_mano_flame_correspondences.zip', zip_path,
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

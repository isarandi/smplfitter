"""smplfitter.download with a fake download server (no network, no account needed)."""

import io
import urllib.error
import urllib.parse
import zipfile

import numpy as np
import pytest

from smplfitter import download as dl
from smplfitter.common import body_models_dir


class FakeServer:
    """Stands in for the opener of the MPI download server; records the requested files."""

    def __init__(self, reject_domains=()):
        self.requests = []
        self.reject_domains = set(reject_domains)

    def open(self, request):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        domain, filename = query['domain'][0], query['sfile'][0]
        self.requests.append((domain, filename))
        if domain in self.reject_domains:
            raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, None)
        return FakeResponse(self.content(filename))

    @staticmethod
    def content(filename):
        if filename == 'SMPL_python_v.1.1.0.zip':
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as zf:
                for gender in ('neutral', 'm', 'f'):
                    zf.writestr(
                        f'SMPL_python_v.1.1.0/smpl/models/basicmodel_{gender}_lbs_10_207_0_v1.1.0.pkl',
                        f'smpl {gender}',
                    )
            return buffer.getvalue()
        if filename.endswith('_kid_template.npy'):
            buffer = io.BytesIO()
            np.save(buffer, np.ones((6890, 3)))
            return buffer.getvalue()
        return f'content of {filename}'.encode()


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(dl, '_make_opener', lambda: fake)
    monkeypatch.setattr(dl, '_download_spin_regressors', lambda smpl_dir: None)  # plain HTTP
    return fake


def test_downloads_only_the_requested_model(server, tmp_path):
    result = dl.download(['smpl'], tmp_path, email='a@b.c', password='pw')
    assert result == tmp_path
    smpl = tmp_path / 'smpl'
    assert (smpl / 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl').read_text() == 'smpl neutral'
    assert np.load(smpl / 'kid_template.npy').shape == (6890, 3)
    assert (smpl / 'SMPL_NEUTRAL.pkl').resolve() == (
        smpl / 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
    ).resolve()
    assert {domain for domain, _ in server.requests} == {'smpl', 'agora'}
    assert not (tmp_path / 'smplx').exists()


def test_present_models_need_no_login(server, tmp_path, monkeypatch):
    dl.download('smpl', tmp_path, email='a@b.c', password='pw')
    n_requests = len(server.requests)

    def no_prompt(*args):
        raise AssertionError('asked for credentials although nothing is missing')

    monkeypatch.setattr('builtins.input', no_prompt)
    monkeypatch.setattr(dl.getpass, 'getpass', no_prompt)
    monkeypatch.delenv('SMPLFITTER_EMAIL', raising=False)
    monkeypatch.delenv('SMPLFITTER_PASSWORD', raising=False)
    dl.download(['smpl'], tmp_path)
    assert len(server.requests) == n_requests


def test_credentials_from_environment(server, tmp_path, monkeypatch):
    monkeypatch.setenv('SMPLFITTER_EMAIL', 'a@b.c')
    monkeypatch.setenv('SMPLFITTER_PASSWORD', 'pw')
    monkeypatch.setattr('builtins.input', lambda *args: pytest.fail('prompted for email'))
    dl.download(['smpl'], tmp_path)
    assert (tmp_path / 'smpl' / 'kid_template.npy').exists()


def test_rejected_login_raises(server, tmp_path):
    server.reject_domains = {'agora'}
    with pytest.raises(PermissionError, match='agora.is.tue.mpg.de'):
        dl.download(['smpl'], tmp_path, email='a@b.c', password='wrong')


def test_no_kid_template_writes_placeholder_without_agora(server, tmp_path):
    dl.download(['smpl'], tmp_path, email='a@b.c', password='pw', kid_template=False)
    kid = np.load(tmp_path / 'smpl' / 'kid_template.npy')
    assert kid.shape == (6890, 3) and not kid.any()
    assert {domain for domain, _ in server.requests} == {'smpl'}


def test_placeholder_is_replaced_by_a_later_full_download(server, tmp_path):
    dl.download(['smpl'], tmp_path, email='a@b.c', password='pw', kid_template=False)
    n_smpl_requests = sum(domain == 'smpl' for domain, _ in server.requests)
    dl.download(['smpl'], tmp_path, email='a@b.c', password='pw')
    assert np.load(tmp_path / 'smpl' / 'kid_template.npy').any()
    # the SMPL files themselves were present, only the kid template was fetched
    assert sum(domain == 'smpl' for domain, _ in server.requests) == n_smpl_requests
    assert ('agora', 'smpl_kid_template.npy') in server.requests


def test_unknown_model_raises(tmp_path):
    with pytest.raises(ValueError, match='smpl-x'):
        dl.download(['smpl-x'], tmp_path)


def test_kid_templates_are_linked_into_variant_folders(tmp_path):
    (tmp_path / 'smpl').mkdir()
    (tmp_path / 'smpl' / 'kid_template.npy').write_bytes(b'kid')
    (tmp_path / 'smplh').mkdir()
    dl._link_kid_templates(tmp_path)
    assert (tmp_path / 'smplh' / 'kid_template.npy').read_bytes() == b'kid'
    assert not (tmp_path / 'smplxlh').exists()


def test_body_models_dir_resolution_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SMPLFITTER_BODY_MODELS', '/a')
    monkeypatch.setenv('DATA_ROOT', '/b')
    assert body_models_dir() == '/a'
    monkeypatch.delenv('SMPLFITTER_BODY_MODELS')
    assert body_models_dir() == '/b/body_models'
    monkeypatch.delenv('DATA_ROOT')
    (tmp_path / 'body_models').mkdir()
    assert body_models_dir() == 'body_models'
    (tmp_path / 'body_models').rmdir()
    assert body_models_dir().endswith('body_models') and 'smplfitter' in body_models_dir()

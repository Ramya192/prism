# ui/uploads.py: the demo notice under every uploader, and the PRISM_SAMPLES_ONLY switch.

import pytest

from ui import uploads


class _FakeSt:
    def __init__(self):
        self.uploader_calls, self.captions, self.infos = [], [], []

    def file_uploader(self, label, **kwargs):
        self.uploader_calls.append((label, kwargs))
        return "WIDGET"

    def caption(self, text):
        self.captions.append(text)

    def info(self, text):
        self.infos.append(text)


@pytest.fixture
def fake(monkeypatch):
    f = _FakeSt()
    monkeypatch.setattr(uploads, "st", f)
    return f


def test_default_shows_uploader_and_notice(fake, monkeypatch):
    monkeypatch.delenv("PRISM_SAMPLES_ONLY", raising=False)
    assert uploads.demo_file_uploader("Upload", type=["csv"], key="k") == "WIDGET"
    assert fake.uploader_calls == [("Upload", {"type": ["csv"], "key": "k"})]
    assert any("Public demo" in c for c in fake.captions)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_samples_only_hides_the_uploader(fake, monkeypatch, value):
    monkeypatch.setenv("PRISM_SAMPLES_ONLY", value)
    assert uploads.demo_file_uploader("Upload", accept_multiple_files=True) == []
    assert uploads.demo_file_uploader("Upload") is None
    assert fake.uploader_calls == []
    assert fake.infos and "sample" in fake.infos[0].lower()


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_other_values_leave_uploads_on(fake, monkeypatch, value):
    monkeypatch.setenv("PRISM_SAMPLES_ONLY", value)
    assert uploads.demo_file_uploader("Upload") == "WIDGET"

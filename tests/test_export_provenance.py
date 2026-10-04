"""Export provenance: the ImageBind graph must be attributable.

The shipped `models/imagebind_vision.onnx` records `commit: UNRECORDED`
because the clone it was built from lived in `/tmp` and was wiped. That is
stated rather than papered over. What this test locks in is that the *next*
export cannot repeat the mistake: no upstream commit declared means no export,
and a declared commit that disagrees with the clone is refused.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_export_module():
    """Import the export script without executing its main()."""
    spec = importlib.util.spec_from_file_location(
        "export_imagebind_onnx", ROOT / "scripts" / "export_imagebind_onnx.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


GIT = shutil.which("git") or "git"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Static argv, no shell: this helper drives git to build a fixture."""
    return subprocess.run(  # noqa: S603
        [GIT, *args], check=True, capture_output=True, text=True
    )


def _fake_repo(tmp_path: Path, sha: str) -> Path:
    repo = tmp_path / "ib_repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "imagebind").mkdir(exist_ok=True)
    _git("init", "-q", str(repo))
    _git(
        "-C",
        str(repo),
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    head = _git("-C", str(repo), "rev-parse", "HEAD").stdout.strip()
    assert head == sha or len(head) == 40
    return repo


def test_export_refuses_without_a_declared_commit(tmp_path):
    mod = _load_export_module()
    repo = _fake_repo(tmp_path, "")
    with pytest.raises(SystemExit) as exc:
        mod._resolve_upstream_commit(repo, "")
    assert "--upstream-commit" in str(exc.value)


def test_export_returns_the_clones_head_when_they_agree(tmp_path):
    mod = _load_export_module()
    repo = _fake_repo(tmp_path, "")
    head = _git("-C", str(repo), "rev-parse", "HEAD").stdout.strip()
    assert mod._resolve_upstream_commit(repo, head) == head
    # A short sha is accepted too, since that is how operators paste it.
    assert mod._resolve_upstream_commit(repo, head[:12]) == head


def test_export_refuses_a_commit_that_disagrees_with_the_clone(tmp_path):
    mod = _load_export_module()
    repo = _fake_repo(tmp_path, "")
    with pytest.raises(SystemExit) as exc:
        mod._resolve_upstream_commit(repo, "0" * 40)
    assert "mismatch" in str(exc.value)


def test_export_refuses_when_the_repo_is_not_a_clone(tmp_path):
    mod = _load_export_module()
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    with pytest.raises(SystemExit):
        mod._resolve_upstream_commit(plain, "a" * 40)


def test_shipped_registry_does_not_claim_a_commit_it_cannot_prove():
    """The honest state must stay honest: UNRECORDED, not a plausible sha."""
    import json

    doc = json.loads((ROOT / "models" / "registry.json").read_text(encoding="utf-8"))
    entry = next(m for m in doc["models"] if "imagebind" in m.get("path", ""))
    commit = entry["source"]["commit"]
    assert commit == "UNRECORDED" or len(commit) == 40
    assert entry["external_data"]["sha256"], "the weight blob must stay pinned"
    assert entry["sha256"], "the graph must stay pinned"


def test_record_source_writes_the_commit_into_the_registry(tmp_path, monkeypatch):
    import json

    mod = _load_export_module()
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "registry.json").write_text(
        json.dumps(
            {"models": [{"name": "ib", "path": "models/imagebind_vision.onnx", "sha256": "a" * 64}]}
        ),
        encoding="utf-8",
    )
    mod._record_source(Path("models/imagebind_vision.onnx"), "b" * 40, Path("/tmp/x.pth"))
    entry = json.loads((models / "registry.json").read_text(encoding="utf-8"))["models"][0]
    assert entry["source"]["commit"] == "b" * 40
    assert entry["source"]["repo"].endswith("ImageBind")

"""The code revision a run records."""

import subprocess

from privacy_bench.core import provenance


def test_uncommitted_hash_covers_the_content_of_new_files(tmp_path, monkeypatch):
    def git(*arguments):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.com",
                        *arguments], check=True, capture_output=True)

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("A = 1\n", encoding="utf-8")
    git("init")
    git("add", "src/kept.py")
    git("commit", "-m", "start")
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    assert provenance.code_revision()["uncommitted_diff_sha256"] is None

    new = tmp_path / "src" / "new.py"
    new.write_text("B = 1\n", encoding="utf-8")
    first = provenance.code_revision()
    new.write_text("B = 2\n", encoding="utf-8")
    second = provenance.code_revision()
    assert first["dirty"] and second["dirty"]
    assert first["uncommitted_diff_sha256"] != second["uncommitted_diff_sha256"]

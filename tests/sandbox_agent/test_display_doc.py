"""Tests for the display_doc tool + LocalFsStorageClient.

Core guarantee under test: display_doc shows a file to the user but its RETURN
VALUE (what goes into the agent's context) never contains the file content — it
fires an out-of-band hook with the content instead.
"""

import asyncio
import os

from sandbox_agent.tools import display_tools as dt
from sandbox_agent.tools.display_tools import DisplayDoc, register_display_hook, unregister_display_hook


def setup_function():
    unregister_display_hook()  # clean slate for this thread


def teardown_function():
    unregister_display_hook()


# --- classification -------------------------------------------------------

def test_classify_by_extension():
    assert dt._classify("a/b/report.md") == "text"
    assert dt._classify("x.PY") == "text"
    assert dt._classify("chart.png") == "image"
    assert dt._classify("photo.JPEG") == "image"
    assert dt._classify("paper.pdf") == "pdf"
    assert dt._classify("model.bin") == "file"
    assert dt._classify("noext") == "file"


# --- the tool -------------------------------------------------------------

def _call(project="proj", path="doc.md"):
    return DisplayDoc().call({"project": project, "path": path})


def test_text_doc_fires_hook_and_return_excludes_content(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    secret = "TOP-SECRET-DOC-BODY " * 50
    (tmp_path / "doc.md").write_text("# Report\n\n" + secret)

    captured = []
    register_display_hook(captured.append)
    result = _call(path="doc.md")

    # hook got the full text (for the UI)
    assert len(captured) == 1
    assert captured[0]["kind"] == "text"
    assert secret in captured[0]["text"]
    assert captured[0]["name"] == "doc.md"
    # but the agent-facing return value does NOT contain the body
    assert secret not in result
    assert "Displayed 'doc.md'" in result
    assert "not in your context" in result.lower()


def test_image_doc_payload_has_no_text(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    (tmp_path / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")
    captured = []
    register_display_hook(captured.append)
    result = _call(path="chart.png")
    assert captured[0]["kind"] == "image"
    assert "text" not in captured[0]            # binary not read into payload
    assert captured[0]["path"].endswith("chart.png")
    assert "chart.png" in result


def test_large_text_downgrades_to_file_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    monkeypatch.setattr(dt, "_INLINE_TEXT_MAX", 100)
    (tmp_path / "big.txt").write_text("x" * 500)
    captured = []
    register_display_hook(captured.append)
    _call(path="big.txt")
    assert captured[0]["kind"] == "file"        # too big to inline
    assert "text" not in captured[0]


def test_path_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    captured = []
    register_display_hook(captured.append)
    out = _call(path="../../etc/passwd")
    assert "Invalid path" in out
    assert captured == []                        # hook never fired


def test_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    out = _call(path="nope.md")
    assert "File not found" in out


def test_no_hook_means_no_chat_surface(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_project_dir", lambda p: str(tmp_path))
    (tmp_path / "doc.md").write_text("hi")
    unregister_display_hook()  # ensure none
    out = _call(path="doc.md")
    assert "no interactive chat surface" in out


# --- LocalFsStorageClient -------------------------------------------------

def test_storage_client_roundtrip(tmp_path, monkeypatch):
    import sandbox_agent.chat_storage as cs
    monkeypatch.setattr(cs, "ELEMENTS_DIR", str(tmp_path / ".cl_elements"))
    client = cs.LocalFsStorageClient()

    key = "user-1/elem-abc"   # Chainlit's "<user>/<elem>" shape (has a slash)
    res = asyncio.run(client.upload_file(key, b"\x00\x01blob", mime="image/png"))
    assert res["object_key"] == key
    assert res["url"] == f"/cl-elements/{key}"
    # blob written under ELEMENTS_DIR with parent dirs created
    written = os.path.join(str(tmp_path / ".cl_elements"), key)
    assert os.path.isfile(written)
    assert open(written, "rb").read() == b"\x00\x01blob"
    # read URL round-trips
    assert asyncio.run(client.get_read_url(key)) == f"/cl-elements/{key}"
    # delete
    assert asyncio.run(client.delete_file(key)) is True
    assert not os.path.exists(written)


def test_storage_client_rejects_traversal(tmp_path, monkeypatch):
    import sandbox_agent.chat_storage as cs
    monkeypatch.setattr(cs, "ELEMENTS_DIR", str(tmp_path / ".cl_elements"))
    import pytest
    with pytest.raises(ValueError):
        cs.resolve_object_path("../../etc/passwd")

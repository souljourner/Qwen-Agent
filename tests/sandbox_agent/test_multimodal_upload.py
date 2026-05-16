"""Tests for the image-upload path: chat_app._build_user_message constructs a
multimodal Message, and qwen_agent.llm.oai._multimodal_to_oai_dict serializes
it to the OpenAI parts wire format that vision-capable vLLM expects."""

import base64
import os
import tempfile

from qwen_agent.llm.oai import _multimodal_to_oai_dict
from qwen_agent.llm.schema import ContentItem, Message
from sandbox_agent.chat_app import _build_user_message, _image_element_to_data_url


# A valid 1x1 transparent PNG — enough bytes for the encoder to produce a real
# data URL without depending on PIL or any external file.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
    "QVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _write_png(tmp_path):
    p = os.path.join(tmp_path, "tiny.png")
    with open(p, "wb") as f:
        f.write(_TINY_PNG)
    return p


class _FakeImageEl:
    """Mimics the bits of cl.Image / cl.Element that _image_element_to_data_url
    inspects (path, mime, content, url)."""
    def __init__(self, *, path=None, content=None, url=None, mime="image/png"):
        self.path = path
        self.content = content
        self.url = url
        self.mime = mime


class _FakeChainlitMsg:
    def __init__(self, content, elements=None):
        self.content = content
        self.elements = elements or []


def test_build_user_message_text_only_returns_plain_string():
    m = _build_user_message(_FakeChainlitMsg("hello"))
    assert m.role == "user" and m.content == "hello"


def test_build_user_message_with_image_returns_multimodal_parts(tmp_path):
    path = _write_png(tmp_path)
    m = _build_user_message(_FakeChainlitMsg(
        "what's in this?", [_FakeImageEl(path=path)]
    ))
    assert m.role == "user"
    assert isinstance(m.content, list) and len(m.content) == 2
    assert m.content[0].text == "what's in this?"
    assert m.content[1].image.startswith("data:image/png;base64,")


def test_build_user_message_falls_back_to_text_if_image_read_fails():
    """If every image element fails to yield a data URL (no path, no content,
    no url), we shouldn't ship an empty multimodal message."""
    m = _build_user_message(_FakeChainlitMsg("hi", [_FakeImageEl()]))
    assert m.content == "hi"


def test_image_element_to_data_url_from_path(tmp_path):
    path = _write_png(tmp_path)
    url = _image_element_to_data_url(_FakeImageEl(path=path))
    assert url and url.startswith("data:image/png;base64,")
    # decodes back to the original bytes
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == _TINY_PNG


def test_image_element_to_data_url_from_inline_bytes():
    url = _image_element_to_data_url(_FakeImageEl(content=_TINY_PNG, mime="image/png"))
    assert url and url.startswith("data:image/png;base64,")


def test_image_element_to_data_url_returns_external_url_as_passthrough():
    """If Chainlit only gave us a URL (no path, no bytes), pass it through —
    vLLM can fetch it if reachable."""
    url = _image_element_to_data_url(_FakeImageEl(url="https://example.com/x.png"))
    assert url == "https://example.com/x.png"


def test_multimodal_to_oai_dict_emits_openai_parts_format():
    msg = Message(role="user", content=[
        ContentItem(text="what's this?"),
        ContentItem(image="data:image/png;base64,XXX"),
    ])
    d = _multimodal_to_oai_dict(msg)
    assert d == {
        "role": "user",
        "content": [
            {"type": "text", "text": "what's this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}},
        ],
    }


def test_multimodal_to_oai_dict_preserves_function_call_and_name():
    """Tool-call messages produced by the agent retain their function_call and
    name fields when run through the multimodal serializer. (Unlikely to be
    hit in practice — assistant tool calls don't carry images — but the
    serializer should be safe to use uniformly.)"""
    msg = Message(role="assistant", name="some_agent",
                  content=[ContentItem(image="data:image/png;base64,Y=")])
    d = _multimodal_to_oai_dict(msg)
    assert d["role"] == "assistant"
    assert d["name"] == "some_agent"
    assert d["content"] == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,Y="}}]

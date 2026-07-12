"""Phase 1 multimodal: attachment plumbing (fake model) + a key-gated real check.

The fake-model tests prove the WIRING — schema, routing, and the exact vision
content-block shape — deterministically and offline. They do NOT prove a real
model reads an image; that claim needs a real backend, so it lives in a single
integration test that skips unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _bootstrap  # noqa: F401  (puts the vendored packages on sys.path)
from prompt_regression import models
from prompt_regression.models import Attachment, attachment_from_bytes, image_content_blocks
from prompt_regression.runner import Case, answer_for


_PNG = base64.b64decode(  # a 1x1 transparent PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


# ---- helpers / builders ----------------------------------------------------

def test_attachment_from_bytes_encodes_base64():
    att = attachment_from_bytes(_PNG, "image/png")
    assert att.kind == "image" and att.media_type == "image/png"
    assert base64.b64decode(att.data) == _PNG


def test_attachment_from_file_infers_media_type(tmp_path):
    p = tmp_path / "probe.png"
    p.write_bytes(_PNG)
    att = models.attachment_from_file(str(p))
    assert att.media_type == "image/png" and base64.b64decode(att.data) == _PNG


def test_attachment_from_file_rejects_unknown_type(tmp_path):
    p = tmp_path / "probe.xyz"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        models.attachment_from_file(str(p))


def test_image_content_blocks_shape():
    att = attachment_from_bytes(_PNG, "image/png")
    blocks = image_content_blocks("what is in this image?", [att])
    # image block(s) first, then the text — the exact Anthropic vision shape
    assert blocks[0] == {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": att.data}}
    assert blocks[-1] == {"type": "text", "text": "what is in this image?"}


def test_image_content_blocks_rejects_audio_for_now():
    audio = Attachment(kind="audio", media_type="audio/wav", data="AA==")
    with pytest.raises(NotImplementedError):
        image_content_blocks("transcribe", [audio])


# ---- routing (fake model spy) ---------------------------------------------

class _FakeVision:
    name = "fake-vision"

    def __init__(self):
        self.saw_prompt = None
        self.saw_attachments = None

    def ask(self, prompt):
        return "TEXT-PATH"

    def ask_multimodal(self, prompt, attachments):
        self.saw_prompt = prompt
        self.saw_attachments = attachments
        return "VISION-PATH"


def _case(**kw):
    base = dict(id="c", category="safety", prompt="p", validator="contains",
                args={"value": "x"}, severity="high")
    base.update(kw)
    return Case(**base)


def test_answer_for_routes_attachments_to_multimodal():
    m = _FakeVision()
    att = attachment_from_bytes(_PNG, "image/png")
    out = answer_for(m, _case(prompt="read this", attachments=(att,)))
    assert out == "VISION-PATH"
    assert m.saw_prompt == "read this" and m.saw_attachments == [att]


def test_answer_for_uses_text_path_without_attachments():
    m = _FakeVision()
    assert answer_for(m, _case()) == "TEXT-PATH"
    assert m.saw_attachments is None


class _TextOnly:
    name = "text-only"
    def ask(self, prompt):
        return "ok"


def test_answer_for_errors_clearly_on_a_blind_backend():
    att = attachment_from_bytes(_PNG, "image/png")
    with pytest.raises(NotImplementedError, match="image/audio"):
        answer_for(_TextOnly(), _case(attachments=(att,)))


# ---- key-gated integration: proves a REAL model reads the image -----------

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                    reason="no ANTHROPIC_API_KEY — real vision integration test skipped")
def test_real_claude_reads_text_in_an_image():
    """OCR sanity check against a real backend. Renders the word HELLO into a
    PNG and asserts Claude reads it back — the thing the fake model cannot prove.
    """
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw
    import io as _io
    from prompt_regression.models import ClaudeModel

    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).text((20, 30), "HELLO", fill="black")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    att = attachment_from_bytes(buf.getvalue(), "image/png")

    reply = ClaudeModel().ask_multimodal("What word is written in this image? Reply with just the word.", [att])
    assert "hello" in reply.lower()

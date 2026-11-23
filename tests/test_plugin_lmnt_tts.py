from __future__ import annotations

import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("lmnt")


@asynccontextmanager
async def _server(body: bytes) -> AsyncIterator[aiohttp.ClientSession]:
    """Local stand-in for the LMNT bytes endpoint that returns ``body`` as is."""

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="application/octet-stream")

    app = web.Application()
    app.router.add_post("/v1/ai/speech/bytes", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    url = f"http://127.0.0.1:{port}/v1/ai/speech/bytes"
    try:
        async with aiohttp.ClientSession() as session:
            with patch("livekit.plugins.lmnt.tts.LMNT_BASE_URL", url):
                yield session
    finally:
        await runner.cleanup()


async def _synthesize(fmt: str, body: bytes) -> bytes:
    from livekit.plugins.lmnt import TTS

    async with _server(body) as session:
        tts = TTS(api_key="test-key", format=fmt, sample_rate=8000, http_session=session)
        stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0, timeout=5))
        audio = bytearray()
        try:
            async for ev in stream:
                audio.extend(ev.frame.data.tobytes())
        finally:
            await stream.aclose()
    return bytes(audio)


async def test_raw_format_is_played_as_pcm():
    # format="raw" returns bare 16-bit little-endian PCM
    pcm = struct.pack("<4h", 1000, -1000, 32767, -32768)
    audio = await _synthesize("raw", pcm)
    assert audio[: len(pcm)] == pcm
    assert set(audio[len(pcm) :]) <= {0}


async def test_mulaw_format_is_decoded_to_pcm():
    # G.711 mu-law: 0xff and 0x7f are zero, 0x80 and 0x00 are the extremes
    audio = await _synthesize("mulaw", bytes([0xFF, 0x7F, 0x80, 0x00]))
    expected = struct.pack("<4h", 0, 0, 32124, -32124)
    assert audio[: len(expected)] == expected
    assert set(audio[len(expected) :]) <= {0}

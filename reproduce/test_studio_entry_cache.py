"""The HTML entry must never reuse an outdated frontend after a server update."""

import asyncio

import httpx

from cofl.app.config import StudioConfig
from cofl.app.server import create_app


def test_entry_is_fresh_even_with_conditional_headers(tmp_path):
    (tmp_path / "index.html").write_text("<html>current studio</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "hashed.js").write_text("// immutable asset")
    app = create_app(StudioConfig(device="cpu"), static_dir=tmp_path)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1") as client:
            first = await client.get("/")
            assert first.status_code == 200
            for entry in ("/", "/index.html", "/?studio_reload=1"):
                result = await client.get(entry, headers={
                    "If-None-Match": first.headers["etag"],
                    "If-Modified-Since": first.headers["last-modified"],
                })
                assert result.status_code == 200
                assert result.headers["cache-control"] == "no-store"
                assert "current studio" in result.text
            asset = await client.get("/assets/hashed.js")
            assert asset.status_code == 200
            assert asset.headers.get("cache-control") != "no-store"

    asyncio.run(check())

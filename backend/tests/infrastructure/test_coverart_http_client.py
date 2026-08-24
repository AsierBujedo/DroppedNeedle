from types import SimpleNamespace

import pytest

from infrastructure.http.client import (
    HttpClientFactory,
    get_coverart_http_client,
    get_http_client,
    get_musicbrainz_http_client,
)


def test_coverart_client_uses_short_budget_and_distinct_name():
    """Covers ride their own short-budget client so a slow archive.org fetch degrades to a
    placeholder instead of holding the request open, and retuning it never touches the shared
    'default' client used by other metadata providers."""
    client = get_coverart_http_client()

    # Short budget: 6s read, 3s connect - not the 10s shared default.
    assert client.timeout.read == 6.0
    assert client.timeout.connect == 3.0

    # Cached under its own name, and a different instance from the default client.
    assert HttpClientFactory._clients.get("coverart") is client
    assert client is not get_http_client()


@pytest.mark.asyncio
async def test_musicbrainz_client_uses_http1_with_configured_limits_and_user_agent():
    settings = SimpleNamespace(
        http_timeout=10.0,
        http_connect_timeout=5.0,
        http_max_connections=50,
        http_max_keepalive=7,
        get_user_agent=lambda: "DroppedNeedle/test (contact@example.test)",
    )
    previous = HttpClientFactory._clients.pop("musicbrainz", None)
    if previous is not None:
        await previous.aclose()

    client = get_musicbrainz_http_client(settings, max_connections=13)
    try:
        pool = client._transport._pool
        assert HttpClientFactory._clients.get("musicbrainz") is client
        assert pool._http2 is False
        assert pool._max_connections == 13
        assert pool._max_keepalive_connections == 7
        assert client.headers["User-Agent"] == settings.get_user_agent()
    finally:
        HttpClientFactory._clients.pop("musicbrainz", None)
        await client.aclose()

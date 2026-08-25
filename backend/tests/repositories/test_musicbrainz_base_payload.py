"""Payload-shape failures from the live MusicBrainz API are data problems, not
service-health signals: ``mb_api_get`` must surface them as
``InvalidExternalPayloadError`` and must never count them toward the shared
circuit breaker. A single release whose payload violates the verified schema
(e.g. an identifier MusicBrainz legitimately sends as JSON null) must not be
able to open the breaker and take the whole integration down.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import infrastructure.resilience.retry as retry_module
import repositories.musicbrainz_base as mb_base
from core.exceptions import ExternalServiceError, InvalidExternalPayloadError
from infrastructure.resilience.retry import CircuitState
from repositories.musicbrainz_management_models import MbManagementRelease


@pytest.fixture
def fake_transport(monkeypatch):
    """Instant limiter and retry sleeps around a pristine shared breaker."""
    monkeypatch.setattr(
        mb_base, "mb_rate_limiter", SimpleNamespace(acquire=AsyncMock())
    )
    monkeypatch.setattr(retry_module, "asyncio", SimpleNamespace(sleep=AsyncMock()))
    mb_base.mb_circuit_breaker.reset()
    yield
    mb_base.mb_circuit_breaker.reset()


def _client(payload: bytes, status: int = 200, calls: list | None = None):
    class _Client:
        async def get(self, url, params=None):
            if calls is not None:
                calls.append(url)
            return httpx.Response(status, content=payload)

    return _Client()


@pytest.mark.asyncio
async def test_payload_schema_mismatch_raises_non_breaking_error(
    fake_transport, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(mb_base, "_http_client", _client(b'{"id": 123}', calls=calls))

    for _ in range(5):
        with pytest.raises(InvalidExternalPayloadError) as captured:
            await mb_base.mb_api_get("/release/x", decode_type=MbManagementRelease)
        assert isinstance(captured.value, ExternalServiceError)

    assert len(calls) == 5
    assert mb_base.mb_circuit_breaker.failure_count == 0
    assert mb_base.mb_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_service_failure_still_counts_toward_breaker(
    fake_transport, monkeypatch
) -> None:
    monkeypatch.setattr(mb_base, "_http_client", _client(b"<html/>", status=503))

    with pytest.raises(ExternalServiceError) as captured:
        await mb_base.mb_api_get("/release/x", decode_type=MbManagementRelease)

    assert type(captured.value) is ExternalServiceError
    assert mb_base.mb_circuit_breaker.failure_count == 1

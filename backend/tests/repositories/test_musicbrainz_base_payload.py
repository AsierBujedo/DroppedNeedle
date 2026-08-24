"""Payload-shape failures from the live MusicBrainz API are data problems, not
service-health signals: ``mb_api_get`` must surface them as
``InvalidExternalPayloadError`` and must never count them toward the shared
circuit breaker. A single release whose payload violates the verified schema
(e.g. an identifier MusicBrainz legitimately sends as JSON null) must not be
able to open the breaker and take the whole integration down.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

import repositories.musicbrainz_base as mb_base
from core.exceptions import ExternalServiceError, InvalidExternalPayloadError
from infrastructure.resilience.retry import CircuitOpenError, CircuitState
from repositories.musicbrainz_management_models import MbManagementRelease


@pytest.fixture
def fake_transport(monkeypatch):
    """Run queue operations and retry sleeps instantly around a pristine breaker."""

    async def execute_immediately(operation, **_kwargs):
        return await operation()

    monkeypatch.setattr(
        mb_base.musicbrainz_request_queue, "execute", execute_immediately
    )
    monkeypatch.setattr(mb_base.asyncio, "sleep", AsyncMock())
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


@pytest.mark.asyncio
async def test_remote_protocol_disconnect_is_not_retried_and_reopens_half_open_breaker(
    fake_transport, monkeypatch
) -> None:
    calls: list[str] = []

    class _ResetThenSuccessClient:
        async def get(self, url, params=None):
            calls.append(url)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError(
                    "<StreamReset stream_id:5, error_code:1, remote_reset:True>"
                )
            return httpx.Response(200, json={"id": "release-x"})

    monkeypatch.setattr(mb_base, "_http_client", _ResetThenSuccessClient())
    mb_base.mb_circuit_breaker.state = CircuitState.HALF_OPEN

    with pytest.raises(httpx.RemoteProtocolError, match="StreamReset"):
        await mb_base.mb_api_get("/release/x")

    assert len(calls) == 1
    assert mb_base.mb_circuit_breaker.state == CircuitState.OPEN
    assert mb_base.mb_circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_429_retries_once_through_queue_after_long_backoff_floor(
    fake_transport, monkeypatch
) -> None:
    queued_attempts: list[int] = []
    responses = [
        httpx.Response(429, headers={"Retry-After": "0.25"}),
        httpx.Response(200, json={"id": "release-x"}),
    ]

    class _RateLimitedThenSuccessClient:
        async def get(self, url, params=None):
            return responses.pop(0)

    async def record_queued_attempt(operation, **kwargs):
        queued_attempts.append(kwargs["attempt"])
        return await operation()

    monkeypatch.setattr(
        mb_base.musicbrainz_request_queue, "execute", record_queued_attempt
    )
    monkeypatch.setattr(mb_base, "_http_client", _RateLimitedThenSuccessClient())

    result = await mb_base.mb_api_get("/release/x")

    assert result == {"id": "release-x"}
    assert queued_attempts == [1, 2]
    mb_base.asyncio.sleep.assert_awaited_once_with(30.0)
    assert mb_base.mb_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_persistent_service_failures_open_breaker(
    fake_transport, monkeypatch
) -> None:
    monkeypatch.setattr(mb_base, "_http_client", _client(b"failure", status=500))

    for _ in range(mb_base.mb_circuit_breaker.failure_threshold):
        with pytest.raises(ExternalServiceError):
            await mb_base.mb_api_get("/release/x")

    assert mb_base.mb_circuit_breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await mb_base.mb_api_get("/release/x")

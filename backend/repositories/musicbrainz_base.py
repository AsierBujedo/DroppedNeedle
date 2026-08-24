import asyncio
import logging
from typing import Any, TypeVar

import httpx
import msgspec

from core.exceptions import (
    ExternalServiceError,
    InvalidExternalPayloadError,
    RateLimitedError,
)
from infrastructure.resilience.retry import (
    CircuitBreaker,
    CircuitOpenError,
)
from infrastructure.resilience.rate_limiter import TokenBucketRateLimiter
from infrastructure.resilience.musicbrainz_queue import musicbrainz_request_queue
from infrastructure.queue.priority_queue import RequestPriority, get_priority_queue
from infrastructure.http.deduplication import RequestDeduplicator
from infrastructure.service_health import report_breaker_health

_mb_api_base: str = "https://musicbrainz.org/ws/2"
logger = logging.getLogger(__name__)

MB_MAX_ATTEMPTS = 2
MB_RETRY_DELAY_SECONDS = 30.0


def get_mb_api_base() -> str:
    return _mb_api_base


def set_mb_api_base(url: str) -> None:
    global _mb_api_base
    _mb_api_base = url.rstrip("/")


mb_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=1,
    timeout=20.0,
    name="musicbrainz",
    on_state_change=report_breaker_health(
        "musicbrainz",
        "metadata",
        message="MusicBrainz, our main source for music data, is having trouble - "
        "search and album or artist details may be incomplete for now.",
    ),
)

# Backward-compatible live settings snapshot. Physical requests do not acquire this
# token bucket: the process-wide FIFO queue below is the authoritative governor.
mb_rate_limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)

mb_deduplicator = RequestDeduplicator()

_http_client: httpx.AsyncClient | None = None
T = TypeVar("T")


def _decode_json_response(response: httpx.Response) -> dict[str, Any]:
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray, memoryview)):
        return msgspec.json.decode(content, type=dict[str, Any])
    return response.json()


def _decode_typed_response(response: httpx.Response, decode_type: type[T]) -> T:
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray, memoryview)):
        return msgspec.json.decode(content, type=decode_type)
    return msgspec.convert(response.json(), type=decode_type)


def set_mb_http_client(client: httpx.AsyncClient) -> None:
    global _http_client
    _http_client = client


def get_mb_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("MusicBrainz HTTP client not initialized")
    return _http_client


async def mb_network_get(
    url: str,
    *,
    params: dict[str, Any],
    priority: RequestPriority,
    label: str,
    attempt: int,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Run one physical MusicBrainz request through the process-wide queue."""

    priority_mgr = get_priority_queue()
    semaphore = await priority_mgr.acquire_slot(priority)
    async with semaphore:
        resolved_client = client or get_mb_http_client()
        return await musicbrainz_request_queue.execute(
            lambda: resolved_client.get(url, params=params),
            label=label,
            attempt=attempt,
        )


async def _mb_api_get_attempt(
    path: str,
    params: dict[str, Any] | None = None,
    priority: RequestPriority = RequestPriority.USER_INITIATED,
    decode_type: type[T] | None = None,
    *,
    attempt: int,
) -> dict[str, Any] | T:
    url = f"{get_mb_api_base()}{path}"
    request_params = dict(params) if params else {}
    request_params["fmt"] = "json"
    response = await mb_network_get(
        url,
        params=request_params,
        priority=priority,
        label=path,
        attempt=attempt,
    )
    if response.status_code == 404:
        if decode_type is not None:
            return decode_type()
        return {}
    if response.status_code == 429:
        retry_after_header = response.headers.get("Retry-After")
        try:
            retry_after = (
                float(retry_after_header) if retry_after_header is not None else None
            )
        except ValueError:
            retry_after = None
        raise RateLimitedError(
            f"MusicBrainz rate limited (429): {path}",
            retry_after_seconds=retry_after,
        )
    if response.status_code == 503:
        raise ExternalServiceError(f"MusicBrainz rate limited (503): {path}")
    if response.status_code != 200:
        raise ExternalServiceError(
            f"MusicBrainz API error ({response.status_code}): {path}"
        )
    try:
        if decode_type is not None:
            return _decode_typed_response(response, decode_type)
        return _decode_json_response(response)
    except msgspec.ValidationError as exc:
        # deterministic per payload (e.g. a field MusicBrainz sends as JSON
        # null), so it says nothing about service health and never counts
        # toward the circuit breaker
        raise InvalidExternalPayloadError(
            f"MusicBrainz returned an unexpected payload shape for {path}: {exc}"
        ) from exc
    except (msgspec.DecodeError, TypeError) as exc:
        raise ExternalServiceError(
            f"MusicBrainz returned invalid JSON payload for {path}: {exc}"
        ) from exc


async def mb_api_get(
    path: str,
    params: dict[str, Any] | None = None,
    priority: RequestPriority = RequestPriority.USER_INITIATED,
    decode_type: type[T] | None = None,
) -> dict[str, Any] | T:
    """Fetch MusicBrainz data without allowing retries to bypass serialization."""

    await mb_circuit_breaker.atry_transition()
    if mb_circuit_breaker.is_open():
        if mb_circuit_breaker.should_log_open_warning():
            logger.warning("Circuit breaker 'musicbrainz' is OPEN")
        raise CircuitOpenError(
            "Circuit breaker 'musicbrainz' is OPEN",
            breaker_name="musicbrainz",
        )

    for attempt in range(1, MB_MAX_ATTEMPTS + 1):
        try:
            result = await _mb_api_get_attempt(
                path,
                params=params,
                priority=priority,
                decode_type=decode_type,
                attempt=attempt,
            )
        except InvalidExternalPayloadError:
            raise
        except httpx.RemoteProtocolError as exc:
            await mb_circuit_breaker.arecord_failure()
            logger.error(
                "MusicBrainz remote protocol failure; not retrying immediately "
                "path=%s attempt=%d error=%s",
                path,
                attempt,
                exc,
            )
            raise
        except (httpx.HTTPError, ExternalServiceError) as exc:
            if attempt >= MB_MAX_ATTEMPTS:
                await mb_circuit_breaker.arecord_failure()
                logger.error(
                    "MusicBrainz request failed after %d serialized attempts "
                    "path=%s error=%s",
                    attempt,
                    path,
                    exc,
                )
                raise
            retry_after = getattr(exc, "retry_after_seconds", None)
            try:
                requested_delay = float(retry_after) if retry_after is not None else 0.0
            except (TypeError, ValueError):
                requested_delay = 0.0
            delay = max(MB_RETRY_DELAY_SECONDS, requested_delay)
            logger.warning(
                "MusicBrainz request will retry through global queue "
                "path=%s next_attempt=%d backoff_seconds=%.1f error=%s",
                path,
                attempt + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
        else:
            await mb_circuit_breaker.arecord_success()
            return result

    raise RuntimeError("MusicBrainz retry loop exited unexpectedly")


def should_include_release(
    release_group: dict[str, Any],
    included_secondary_types: set[str] | None = None,
    included_primary_types: set[str] | None = None,
) -> bool:
    if included_primary_types is not None:
        primary_type = (release_group.get("primary-type") or "").lower()
        if primary_type not in included_primary_types:
            return False

    secondary_types = set(
        map(str.lower, release_group.get("secondary-types", []) or [])
    )

    if included_secondary_types is None:
        exclude_types = {
            "compilation",
            "live",
            "remix",
            "soundtrack",
            "dj-mix",
            "mixtape/street",
            "demo",
        }
        return secondary_types.isdisjoint(exclude_types)

    if not secondary_types:
        return "studio" in included_secondary_types

    return bool(secondary_types.intersection(included_secondary_types))


def extract_artist_name(release_group: dict[str, Any]) -> str | None:
    artist_credit = release_group.get("artist-credit", [])
    if not isinstance(artist_credit, list) or not artist_credit:
        return None

    first_credit = artist_credit[0]
    if isinstance(first_credit, dict):
        return first_credit.get("name") or (first_credit.get("artist") or {}).get(
            "name"
        )
    return None


def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    year = date_str.split("-", 1)[0]
    return int(year) if year.isdigit() else None


def get_score(item: dict[str, Any]) -> int:
    score = item.get("score") or item.get("ext:score")
    try:
        return int(score) if score else 0
    except (ValueError, TypeError):
        return 0


def dedupe_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    for item in items:
        item_id = item.get("id")
        if item_id and item_id not in seen:
            seen[item_id] = item

    result = list(seen.values())
    result.sort(key=get_score, reverse=True)
    return result


def _normalize_tag_phrase(tag: str) -> str:
    return " ".join(tag.strip().lower().split())


_LUCENE_RESERVED = frozenset(r'+-&|!(){}[]^"~*?:\\/')


def escape_lucene_phrase(value: str) -> str:
    """Escape user text before placing it inside a Lucene field phrase."""

    return "".join(
        f"\\{character}" if character in _LUCENE_RESERVED else character
        for character in value
    )


def build_release_search_query(title: str, artist: str) -> str:
    """Build a release query live-verified against MusicBrainz WS/2 on 2026-08-13."""

    clauses = [f'release:"{escape_lucene_phrase(title)}"']
    if artist:
        clauses.append(f'artist:"{escape_lucene_phrase(artist)}"')
    return " AND ".join(clauses)


def build_release_group_search_query(title: str, artist: str) -> str:
    """Build a release-group query live-verified against MusicBrainz WS/2 on 2026-08-13."""

    escaped_title = escape_lucene_phrase(title)
    query = f'(releasegroup:"{escaped_title}" OR release:"{escaped_title}")'
    if artist:
        query += f' AND artist:"{escape_lucene_phrase(artist)}"'
    return query


def build_recording_search_query(title: str, artist: str) -> str:
    """Build a recording query using the same verified Lucene field escaping."""

    return (
        f'recording:"{escape_lucene_phrase(title)}" AND '
        f'artist:"{escape_lucene_phrase(artist)}"'
    )


def _escape_tag_phrase(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_musicbrainz_tag_query(tag: str) -> str:
    base = _normalize_tag_phrase(tag)
    if not base:
        return 'tag:""^3'

    variants: list[str] = [base]
    seen = {base}

    def add_variant(value: str) -> None:
        normalized = _normalize_tag_phrase(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(normalized)

    add_variant(base.replace("-", " "))
    add_variant(base.replace(" ", "-"))

    if "&" in base:
        add_variant(base.replace("&", " and "))
        add_variant(base.replace("&", " "))

    if " and " in base:
        add_variant(base.replace(" and ", " & "))
        add_variant(base.replace(" and ", " "))

    clauses = []
    for index, variant in enumerate(variants):
        escaped = _escape_tag_phrase(variant)
        boost = "^3" if index == 0 else "^2"
        clauses.append(f'tag:"{escaped}"{boost}')

    return " OR ".join(clauses)

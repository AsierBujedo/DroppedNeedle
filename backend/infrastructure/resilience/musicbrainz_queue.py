"""Process-wide serialization for every physical MusicBrainz request."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
MIN_MUSICBRAINZ_INTERVAL_SECONDS = 1.5


class MusicBrainzRequestQueue:
    """FIFO gate with one in-flight request and a minimum start interval.

    ``asyncio.Lock`` wakes waiters in acquisition order. The additional state lock is
    used only for observable queue depth and cancellation-safe accounting; the serial
    lock remains held across the spacing delay and the complete HTTP operation.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if min_interval_seconds < MIN_MUSICBRAINZ_INTERVAL_SECONDS:
            raise ValueError(
                "MusicBrainz request interval must be at least 1.5 seconds"
            )
        self.min_interval_seconds = float(min_interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._serial_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._queued = 0
        self._active = False
        self._last_attempt_started: float | None = None
        self._sequence = 0

    def update_rate(self, requests_per_second: float) -> None:
        """Honor slower configured rates without relaxing the 1.5-second floor."""

        if requests_per_second <= 0:
            raise ValueError("MusicBrainz request rate must be positive")
        self.min_interval_seconds = max(
            MIN_MUSICBRAINZ_INTERVAL_SECONDS,
            1.0 / float(requests_per_second),
        )
        logger.info(
            "MusicBrainz queue configured requests_per_second=%.3f "
            "min_interval_seconds=%.3f max_in_flight=1",
            requests_per_second,
            self.min_interval_seconds,
        )

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        label: str,
        attempt: int,
    ) -> T:
        enqueued_at = self._clock()
        async with self._state_lock:
            self._queued += 1
            self._sequence += 1
            request_number = self._sequence
            position = self._queued
            active = self._active

        logger.info(
            "MusicBrainz queue enqueued request=%d attempt=%d path=%s "
            "position=%d in_flight=%s",
            request_number,
            attempt,
            label,
            position,
            active,
        )

        acquired = False
        try:
            await self._serial_lock.acquire()
            acquired = True
            async with self._state_lock:
                self._queued -= 1
                self._active = True
                waiting = self._queued

            now = self._clock()
            spacing_delay = 0.0
            if self._last_attempt_started is not None:
                spacing_delay = max(
                    0.0,
                    self.min_interval_seconds - (now - self._last_attempt_started),
                )
            if spacing_delay > 0:
                logger.info(
                    "MusicBrainz queue spacing request=%d attempt=%d path=%s "
                    "wait_seconds=%.3f queued_behind=%d",
                    request_number,
                    attempt,
                    label,
                    spacing_delay,
                    waiting,
                )
                await self._sleep(spacing_delay)

            started_at = self._clock()
            self._last_attempt_started = started_at
            logger.info(
                "MusicBrainz queue dispatch request=%d attempt=%d path=%s "
                "queue_wait_seconds=%.3f queued_behind=%d",
                request_number,
                attempt,
                label,
                max(0.0, started_at - enqueued_at),
                waiting,
            )

            outcome = "error"
            try:
                result = await operation()
                outcome = "success"
                return result
            finally:
                logger.info(
                    "MusicBrainz queue finished request=%d attempt=%d path=%s "
                    "outcome=%s duration_seconds=%.3f",
                    request_number,
                    attempt,
                    label,
                    outcome,
                    max(0.0, self._clock() - started_at),
                )
        finally:
            if acquired:
                async with self._state_lock:
                    self._active = False
                self._serial_lock.release()
            else:
                async with self._state_lock:
                    self._queued -= 1
                logger.info(
                    "MusicBrainz queue cancelled request=%d attempt=%d path=%s",
                    request_number,
                    attempt,
                    label,
                )


musicbrainz_request_queue = MusicBrainzRequestQueue()

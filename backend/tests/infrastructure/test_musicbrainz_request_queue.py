import asyncio
import logging

import pytest

from infrastructure.resilience.musicbrainz_queue import MusicBrainzRequestQueue


@pytest.mark.asyncio
async def test_release_group_and_release_share_one_fifo_lane_with_spacing(caplog):
    now = 0.0
    real_sleep = asyncio.sleep

    def clock() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        now += delay
        await real_sleep(0)

    queue = MusicBrainzRequestQueue(
        min_interval_seconds=1.5,
        clock=clock,
        sleep=fake_sleep,
    )
    first_release = asyncio.Event()
    active = 0
    max_active = 0
    starts: list[tuple[str, float]] = []

    async def operation(label: str, *, block: bool = False) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        starts.append((label, clock()))
        if block:
            await first_release.wait()
        active -= 1
        return label

    caplog.set_level(logging.INFO)
    group_task = asyncio.create_task(
        queue.execute(
            lambda: operation("release-group", block=True),
            label="/release-group/rg",
            attempt=1,
        )
    )
    await real_sleep(0)
    release_task = asyncio.create_task(
        queue.execute(
            lambda: operation("release"),
            label="/release/release",
            attempt=1,
        )
    )
    await real_sleep(0)

    assert max_active == 1
    first_release.set()
    assert await asyncio.gather(group_task, release_task) == [
        "release-group",
        "release",
    ]
    assert max_active == 1
    assert starts == [("release-group", 0.0), ("release", 1.5)]
    assert "position=1 in_flight=True" in caplog.text
    assert "wait_seconds=1.500" in caplog.text
    assert "queue_wait_seconds=1.500" in caplog.text


def test_queue_rejects_interval_below_musicbrainz_floor():
    with pytest.raises(ValueError, match="at least 1.5 seconds"):
        MusicBrainzRequestQueue(min_interval_seconds=1.49)


def test_configured_rate_can_slow_but_never_accelerate_queue():
    queue = MusicBrainzRequestQueue()

    queue.update_rate(0.5)
    assert queue.min_interval_seconds == 2.0

    queue.update_rate(50.0)
    assert queue.min_interval_seconds == 1.5

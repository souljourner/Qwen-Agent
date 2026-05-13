"""Tests for sandbox_agent.model_tracker — focused on the preview-write
timestamp that cron_loop._is_progressing uses as a stuck-detector signal."""

import time

from sandbox_agent import model_tracker as mt


def setup_function():
    mt.clear_agent_status()


def test_get_last_preview_at_is_none_until_first_write():
    assert mt.get_last_preview_at() is None


def test_set_current_preview_advances_timestamp():
    before = mt.get_last_preview_at()
    mt.set_current_preview("epoch 1 done")
    after = mt.get_last_preview_at()
    assert before is None
    assert after is not None  # advanced


def test_set_current_preview_none_clears_timestamp():
    mt.set_current_preview("hello")
    assert mt.get_last_preview_at() is not None
    mt.set_current_preview(None)  # explicit clear bypasses throttle
    assert mt.get_last_preview_at() is None


def test_preview_throttle_skips_writes_but_repeated_calls_eventually_advance():
    """The internal 1s throttle drops writes within the interval; once it
    elapses, the next write advances the timestamp again."""
    mt.set_current_preview("first")
    first = mt.get_last_preview_at()
    assert first is not None
    # Immediate second call — throttled, timestamp unchanged.
    mt.set_current_preview("second")
    assert mt.get_last_preview_at() == first
    # Wait for the throttle to release, then write again.
    time.sleep(1.05)
    mt.set_current_preview("third")
    third = mt.get_last_preview_at()
    assert third is not None and third > first

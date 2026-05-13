"""Tests for the hysteresis state machine."""

from whospeaks.addon.state_machine import (
    STATE_IDLE,
    STATE_JEROEN,
    STATE_OTHER,
    HysteresisFSM,
)


def test_starts_idle_and_ignores_feeds():
    fsm = HysteresisFSM()
    assert fsm.state == STATE_IDLE
    t = fsm.feed(STATE_JEROEN)
    assert t.committed_state == STATE_IDLE
    assert t.changed is False


def test_start_streaming_enters_other():
    fsm = HysteresisFSM()
    t = fsm.start_streaming()
    assert t.committed_state == STATE_OTHER
    assert t.changed is True
    assert fsm.state == STATE_OTHER


def test_enter_requires_three_consecutive_jeroen():
    fsm = HysteresisFSM()
    fsm.start_streaming()

    t1 = fsm.feed(STATE_JEROEN)
    t2 = fsm.feed(STATE_JEROEN)
    assert t1.committed_state == STATE_OTHER and t1.changed is False
    assert t2.committed_state == STATE_OTHER and t2.changed is False

    t3 = fsm.feed(STATE_JEROEN)
    assert t3.committed_state == STATE_JEROEN
    assert t3.changed is True


def test_jeroen_streak_reset_by_other():
    fsm = HysteresisFSM()
    fsm.start_streaming()
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    # interrupted by OTHER — streak resets
    fsm.feed(STATE_OTHER)
    t = fsm.feed(STATE_JEROEN)
    assert t.committed_state == STATE_OTHER  # only 1 jeroen since reset


def test_leave_requires_two_consecutive_other():
    fsm = HysteresisFSM()
    fsm.start_streaming()
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    assert fsm.state == STATE_JEROEN

    t1 = fsm.feed(STATE_OTHER)
    assert t1.committed_state == STATE_JEROEN and t1.changed is False

    t2 = fsm.feed(STATE_OTHER)
    assert t2.committed_state == STATE_OTHER and t2.changed is True


def test_other_streak_reset_by_jeroen_while_committed_to_jeroen():
    fsm = HysteresisFSM()
    fsm.start_streaming()
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_OTHER)
    # Single JEROEN in the middle should reset the OTHER streak.
    fsm.feed(STATE_JEROEN)
    t = fsm.feed(STATE_OTHER)
    assert t.committed_state == STATE_JEROEN
    assert t.changed is False


def test_reset_to_idle_clears_counters():
    fsm = HysteresisFSM()
    fsm.start_streaming()
    fsm.feed(STATE_JEROEN)
    fsm.feed(STATE_JEROEN)
    fsm.reset(STATE_IDLE)
    assert fsm.state == STATE_IDLE
    # After re-entering streaming, no leftover JEROEN votes count.
    fsm.start_streaming()
    t = fsm.feed(STATE_JEROEN)
    assert t.committed_state == STATE_OTHER

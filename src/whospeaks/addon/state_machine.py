"""Hysteresis state machine for committed speaker classifications.

Asymmetric: harder to enter JEROEN_VAN_INKEL than to leave, compounding with
the model's precision-tuned threshold. See docs/home-assistant-addon.md.
"""

from dataclasses import dataclass

STATE_JEROEN = "JEROEN_VAN_INKEL"
STATE_OTHER = "OTHER"
STATE_IDLE = "idle"
STATE_UNAVAILABLE = "unavailable"

K_ENTER = 3
K_LEAVE = 2


@dataclass
class Transition:
    """Result of feeding a raw label.

    `committed_state` is the new committed state. `changed` indicates whether
    it differs from the previous committed state — callers use this to decide
    whether to publish a state-topic update vs. just an attributes update.
    """

    committed_state: str
    changed: bool


class HysteresisFSM:
    def __init__(self, k_enter: int = K_ENTER, k_leave: int = K_LEAVE):
        self._k_enter = k_enter
        self._k_leave = k_leave
        self._state = STATE_IDLE
        self._consec_jeroen = 0
        self._consec_other = 0

    @property
    def state(self) -> str:
        return self._state

    def reset(self, to: str = STATE_IDLE) -> Transition:
        """Hard reset; used on station change, disconnect, or pause."""
        changed = self._state != to
        self._state = to
        self._consec_jeroen = 0
        self._consec_other = 0
        return Transition(committed_state=self._state, changed=changed)

    def feed(self, raw_label: str) -> Transition:
        """Feed a raw per-cycle classification and return the committed state.

        While in `idle` or `unavailable`, raw predictions are ignored — the
        machine only classifies once a tappable stream is live, which is
        signaled by calling `start_streaming()`.
        """
        if self._state in (STATE_IDLE, STATE_UNAVAILABLE):
            return Transition(committed_state=self._state, changed=False)

        if raw_label == STATE_JEROEN:
            self._consec_jeroen += 1
            self._consec_other = 0
        elif raw_label == STATE_OTHER:
            self._consec_other += 1
            self._consec_jeroen = 0
        else:
            raise ValueError(f"unexpected raw_label {raw_label!r}")

        prev = self._state
        if self._state == STATE_OTHER and self._consec_jeroen >= self._k_enter:
            self._state = STATE_JEROEN
        elif self._state == STATE_JEROEN and self._consec_other >= self._k_leave:
            self._state = STATE_OTHER

        return Transition(committed_state=self._state, changed=self._state != prev)

    def start_streaming(self) -> Transition:
        """Called when a tappable station starts; enters classifier in OTHER."""
        prev = self._state
        self._state = STATE_OTHER
        self._consec_jeroen = 0
        self._consec_other = 0
        return Transition(committed_state=self._state, changed=self._state != prev)

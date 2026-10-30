"""When the microphone may be hearing CLAUDIO itself.

Every sound of CLAUDIO's own (a beep, a reply, an announcement) reaches the microphone while
it plays and for a while after the player has handed it over: the output device still has
part of it queued (more with HDMI or Bluetooth speakers) and the room keeps its echo. The
wake word and the windows that open after CLAUDIO spoke do not listen then (JenJo 28/10: the
beep that closes a conversation woke the "claudio" model again and again).

Time is counted in microphone samples, the clock every listener lives by: a frame is 80 ms
of the room whenever it is processed."""

from __future__ import annotations


class OwnSound:
    """`tail_ms`: how long the room and the microphone path keep a sound after it came out
    of the speakers ([conversation] echo_tail_ms)."""

    def __init__(self, sample_rate: int, tail_ms: int = 0) -> None:
        self.sample_rate = sample_rate
        self.tail_ms = tail_ms
        self._playing = 0  # sounds being played right now
        self._left = 0  # samples after the last sound that may still carry it

    @property
    def playing(self) -> bool:
        return self._playing > 0

    def started(self) -> None:
        """A sound of CLAUDIO's own begins."""
        self._playing += 1

    def ended(self, queued_s: float = 0.0) -> None:
        """A sound was handed to the player: `queued_s` of it has still to come out of the
        speakers, then the room echoes it for tail_ms."""
        self._playing = max(self._playing - 1, 0)
        tail_s = max(queued_s, 0.0) + self.tail_ms / 1000
        self._left = max(self._left, round(tail_s * self.sample_rate))

    def in_frame(self, samples: int) -> bool:
        """One frame of `samples` from the microphone: True when CLAUDIO's own sound may be
        in it. Called once per frame in every state, so the tail runs in real time."""
        own = self._playing > 0 or self._left > 0
        self._left = max(self._left - samples, 0)
        return own

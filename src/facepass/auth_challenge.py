"""Fresh randomized blink/head-turn challenge; experimental RGB anti-replay."""

import math
import secrets


class ActiveChallenge:
    def __init__(self, clock, *, rng=None):
        self.clock = clock
        self.rng = rng or secrets.SystemRandom()
        actions = [("blink", self.rng.choice((1, 2))), ("turn", -1),
                   ("blink", self.rng.choice((1, 2))), ("turn", 1)]
        self.rng.shuffle(actions)
        self.actions = actions
        self.started = clock()
        self.stage = -1
        self.center_yaw = None
        self.hold_started = None
        self.blink_baseline = None
        self.returning = False
        self.done = False
        self.failed = False
        self.last_timestamp = None
        self.track_id = None
        self.last_blink_count = 0

    @property
    def prompt(self):
        if self.failed:
            return "FacePass could not verify you. Use your password or retry."
        if self.done:
            return "FacePass verified."
        if self.stage < 0 or self.returning:
            return "FacePass: face the camera and keep your eyes open."
        action, amount = self.actions[self.stage]
        if action == "blink":
            count = max(1, amount - (self.last_blink_count - self.blink_baseline))
            return f"FacePass: blink slowly {count} {'time' if count == 1 else 'times'}."
        direction = "right" if amount < 0 else "left"
        return f"FacePass: turn your head slightly to your {direction}."

    def _advance(self, blink_count):
        self.stage += 1
        self.hold_started = None
        self.blink_baseline = blink_count
        self.returning = False
        if self.stage == len(self.actions):
            self.done = True

    def update(self, timestamp, *, track_id, yaw, blink_count, eyes_open, identity_match,
               quality_ok, conflict=False):
        if self.done or self.failed:
            return False
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or self.clock() - self.started > 25 or timestamp > self.clock() + .1
                or self.clock() - timestamp > .35
                or (self.last_timestamp is not None and not 0 < timestamp - self.last_timestamp <= .35)
                or (self.track_id is not None and track_id != self.track_id)
                or type(track_id) is not int or track_id < 1
                or type(eyes_open) is not bool or type(identity_match) is not bool or type(quality_ok) is not bool
                or conflict or not quality_ok or not identity_match
                or type(blink_count) is not int or blink_count < self.last_blink_count):
            self.failed = True
            return False
        self.last_timestamp = timestamp
        self.track_id = track_id
        self.last_blink_count = blink_count
        if yaw is None or type(yaw) not in (float, int) or not math.isfinite(yaw) or abs(yaw) > 45:
            self.hold_started = None
            return False
        if self.stage < 0:
            if eyes_open and abs(yaw) < 15:
                if self.hold_started is None:
                    self.hold_started = timestamp
                if timestamp - self.hold_started >= .3:
                    self.center_yaw = yaw
                    self._advance(blink_count)
            else:
                self.hold_started = None
            return False
        delta = yaw - self.center_yaw
        if self.returning:
            condition = eyes_open and abs(delta) <= 7
        else:
            action, amount = self.actions[self.stage]
            if action == "blink":
                if eyes_open and blink_count - self.blink_baseline >= amount:
                    self._advance(blink_count)
                return self.done
            condition = eyes_open and delta * amount >= 15
        if condition:
            if self.hold_started is None:
                self.hold_started = timestamp
            if timestamp - self.hold_started >= .2:
                if self.returning:
                    self._advance(blink_count)
                else:
                    self.returning = True
                    self.hold_started = None
        else:
            self.hold_started = None
        return self.done

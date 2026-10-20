# Unit tests for grasp.close_and_check's outcome logic, with a fake robot (no ROS needed).
from types import SimpleNamespace

from pill_bottle_nodes import grasp

PLAN = SimpleNamespace(lift=0.75, lift_clear=0.9, arm_grasp=0.3, arm_pregrasp=0.2, base_translate=0.0, wrist_yaw=0.0)
S = dict(grasp.GRASP_SETTINGS, hold_test_s=0.3)


class FakeClaw:
    """finger_seq: finger readings returned in order (last one repeats)."""
    def __init__(self, finger_seq):
        self.seq, self.moves = list(finger_seq), []

    def snapshot(self, *a, **k):
        return False

    def finger(self):
        v = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
        return v, 1.0

    def wait_for(self, *a, **k):
        return None


def run(seq):
    c = FakeClaw(seq)
    log = {}
    moves = []
    out = grasp.close_and_check(c, PLAN, S, lambda label, **j: moves.append(j), lambda t: None, log)
    return out, log, moves


def test_missed_when_fingers_close_fully():
    out, log, moves = run([0.0])
    assert out == "missed" and log["finger_closed"] == 0.0
    assert {"joint_lift"} not in [set(m) for m in moves]       # no lift after a miss


def test_held_when_finger_steady():
    out, log, moves = run([0.12, 0.12, 0.121])
    assert out == "held"
    assert moves[0] == {"wrist_extension": PLAN.arm_grasp + S["extra_reach"]}
    assert moves[1] == {"gripper_aperture": S["gripper_squeeze"]}
    assert moves[2] == {"joint_lift": PLAN.lift + S["lift_test"]}


def test_slipping_when_finger_drifts():
    out, _, _ = run([0.12, 0.11, 0.09])
    assert out == "slipping"


def test_dropped_when_finger_closes_while_lifting():
    out, _, _ = run([0.12, 0.05, 0.0])
    assert out == "dropped"


def test_reach_is_clamped():
    plan = SimpleNamespace(**dict(vars(PLAN), arm_grasp=0.499))
    c = FakeClaw([0.12])
    moves = []
    grasp.close_and_check(c, plan, S, lambda label, **j: moves.append(j), lambda t: None, {})
    assert moves[0]["wrist_extension"] == S["arm_max"]

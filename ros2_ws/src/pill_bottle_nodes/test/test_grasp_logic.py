# Unit tests for grasp.close_and_check's outcome logic, with a fake robot (no ROS needed).
# Effort values match what test_gripper measured on the robot: about -0.1 empty, -6.6 on the bottle.
from types import SimpleNamespace

from pill_bottle_nodes import grasp

PLAN = SimpleNamespace(lift=0.75, lift_clear=0.9, arm_grasp=0.35, arm_pregrasp=0.2, base_translate=0.0, wrist_yaw=0.0,
                       grasp_depth=0.05)
S = dict(grasp.GRASP_SETTINGS, hold_test_s=0.3)


class FakeClaw:
    """effort_seq: gripper effort readings returned in order (last one repeats). Finger is always -0.028."""
    def __init__(self, effort_seq):
        self.seq = list(effort_seq)

    def snapshot(self, *a, **k):
        return False

    def finger(self):
        v = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
        return -0.028, v

    def wait_for(self, *a, **k):
        return None


def run(seq, plan=PLAN):
    c = FakeClaw(seq)
    log, moves = {}, []
    out = grasp.close_and_check(c, plan, S, lambda label, **j: moves.append(j), lambda t: None, log)
    return out, log, moves


def test_missed_when_effort_stays_low():
    out, log, moves = run([-0.1])
    assert out == "missed" and log["effort_closed"] == -0.1
    assert all("joint_lift" not in m for m in moves)          # no lift after a miss


def test_held_when_effort_stays_high():
    out, _, moves = run([-6.6, -6.5, -6.4])
    assert out == "held"
    assert moves[0] == {"wrist_extension": PLAN.arm_grasp}      # planner already added grasp_depth
    assert moves[1] == {"gripper_aperture": S["gripper_squeeze"]}
    assert moves[2] == {"joint_lift": PLAN.lift + S["lift_test"]}


def test_slipping_when_effort_sags():
    out, _, _ = run([-6.6, -3.1])                               # above the empty level, below half the grip
    assert out == "slipping"


def test_dropped_when_effort_falls_to_empty():
    out, _, _ = run([-6.6, -2.0, -0.1])
    assert out == "dropped"


def test_same_finger_position_does_not_mean_missed():
    # the case from the robot: identical finger position empty and holding, only effort differs
    assert run([-6.6])[0] == "held"
    assert run([-0.1])[0] == "missed"


def test_reach_is_clamped():
    plan = SimpleNamespace(**dict(vars(PLAN), arm_grasp=0.6))
    _, _, moves = run([-6.6], plan)
    assert moves[0]["wrist_extension"] == S["arm_max"]

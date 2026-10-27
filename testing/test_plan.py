"""
test_plan.py - every component test from the GitHub tickets, in one place.

T1a, T1b, T2, T3 and T6 have automatic ROS testers (listed under "auto").
run_manual_test.py walks you through the others (T4, T8, T9, T10, T11) trial by trial.
Edit conditions / trials / metrics here when the plan changes.
"""

TESTS = {
    "T1a": {
        "name": "Find the bottle and plan the pick (head only, no base/arm motion)",
        "auto": "ros2 run bottle_tracking t1a_plan_tester --ros-args --params-file $P",
        "setup": "Head-only search (base never turns). Prints PLAN + the plan, or NO PLAN + why + how to fix.",
        "conditions": ["any_spot"],
        "trials": 3,
        "success": "Bottle in view -> PLAN printed. Saves a picture of every planned approach.",
        "outcomes": ["SUCCESS", "NOT_FOUND", "NO_PLAN", "ERROR"],
        "metrics": ["search_time_s", "distance_m", "base_steps", "arm_extension_m"],
    },
    "T1b": {
        "name": "Find the bottle, plan, drive and grab it",
        "auto": "ros2 run bottle_tracking t1b_execute_tester --ros-args --params-file $P",
        "setup": "Same as T1a, then the robot drives, re-checks up close, and grabs.",
        "conditions": ["any_spot"],
        "trials": 3,
        "success": "No collision, middle-of-finger grip, lifts and holds 5 s.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "NO_PLAN", "COLLISION", "DROP", "ERROR"],
        "metrics": ["exec_time_s", "final_off_arm_line_cm", "failed_step", "grip_location"],
    },
    "T2": {
        "name": "Find the pill bottle with the head camera",
        "auto": "ros2 run bottle_tracking t2_find_tester --ros-args --params-file $P",
        "setup": "Pause, move the bottle, head-only search. Reports position, distance, and whether it must move.",
        "conditions": ["any_spot"],
        "trials": 5,
        "success": "Finds the right object (not a distractor) with under 2 cm spread.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "FALSE_POSITIVE", "ERROR"],
        "metrics": ["search_time_s", "distance_m", "spread_cm", "confidence"],
    },
    "T3": {
        "name": "Keep an eye on the bottle while moving, then remember it",
        "auto": "ros2 run bottle_tracking t3_tracking_tester --ros-args --params-file $P",
        "setup": "Part A: moves around with tracking on. Part B: moves with the head frozen, then looks back.",
        "conditions": ["A_turn_45", "A_turn_90", "A_drive", "A_mixed", "B_rot+90", "B_rot-90", "B_rot180",
                       "B_drive+0.3"],
        "trials": 3,
        "success": "A: seen >=90% of the time, error <=10 deg, drift <=3 cm. "
                   "B: looks back at the bottle in <=3 s, drift <=3 cm.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["percent_time_seen", "max_tracking_error_deg", "reacquire_time_s", "position_drift_cm"],
    },
    "T4": {
        "name": "'Nothing there' recovery when the bottle disappears",
        "setup": "Robot finds the bottle, then a person removes it at the stage below.",
        "conditions": ["removed_before_driving", "removed_before_grasp", "removed_while_extending"],
        "trials": 3,
        "success": "No grasp at the empty spot; if extending, stops and retracts without closing; "
                   "back to searching in <=10 s; no crash.",
        "outcomes": ["SUCCESS", "MISS", "TIMEOUT", "ERROR"],
        "metrics": ["recovery_time_s", "gripper_closed"],
    },
    "T6": {
        "name": "Distance to the pill bottle (base still)",
        "auto": "ros2 run bottle_tracking t6_distance_tester --ros-args --params-file $P",
        "setup": "Base never moves. 60 s pause between spots. Optional tape-measured distance for the error.",
        "conditions": ["any_spot"],
        "trials": 5,
        "success": "Steady within 1 cm; within 2 cm of the tape (3 cm past 1.2 m).",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["est_m", "true_m", "error_cm", "spread_cm"],
    },
    "T8": {
        "name": "Grasp only (gripper isolated)",
        "setup": "Place the gripper around the bottle directly; test close -> lift -> hold.",
        "conditions": ["effort_low_middle", "effort_med_middle", "effort_high_middle", "effort_med_near_cap"],
        "trials": 5,
        "success": "Middle of fingers contacts bottle, lifts >=5 cm, holds 5 s, not crushed.",
        "outcomes": ["SUCCESS", "MISS", "DROP", "ERROR"],
        "metrics": ["gripper_value", "grasp_height_cm"],
    },
    "T9": {
        "name": "Bottle orientation classification",
        "setup": "Log the TRUE orientation as the condition.",
        "conditions": ["upright", "lying_down"],
        "trials": 5,
        "success": "Predicted orientation matches true orientation.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["predicted_orientation", "confidence"],
    },
    "T10": {
        "name": "In-hand verification at the head camera",
        "setup": "Part A bottle in gripper, Part B empty gripper.",
        "conditions": ["A_bottle_in_gripper", "B_empty_gripper"],
        "trials": 5,
        "success": "A: confirms pill_bottle in <=5 s. B: verification correctly fails.",
        "outcomes": ["SUCCESS", "MISS", "FALSE_POSITIVE", "TIMEOUT", "ERROR"],
        "metrics": ["verify_time_s", "confidence"],
    },
    "T11": {
        "name": "Retry and failure handling",
        "setup": "Take the bottle away during the grasp to force a failed grab.",
        "conditions": ["fail_once_then_succeed", "fail_every_time"],
        "trials": 3,
        "success": "Re-detects before each retry, max 3 retries (4 attempts), then reports, stows, stops.",
        "outcomes": ["SUCCESS", "MISS", "TIMEOUT", "ERROR"],
        "metrics": ["attempts", "final_state"],
    },
}

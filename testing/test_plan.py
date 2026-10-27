"""
test_plan.py - every component test from the GitHub tickets, in one place.

run_manual_test.py reads this to walk you through any test trial by trial.
Tests that have an automatic tester (T3, T5, T12, T15a, T15b) list it under "auto";
you can still log them by hand with run_manual_test.py if you want.

Edit conditions / trials / metrics here when the plan changes.
"""

TESTS = {
    "T1a": {
        "name": "Grasp planning from known coordinates (no motion)",
        "setup": "Planner in dry-run mode. Type in the bottle pose. Robot must not move.",
        "conditions": ["reach_pose1", "reach_pose2", "reach_pose3", "reach_pose4", "reach_pose5",
                       "unreachable_far", "unreachable_low"],
        "trials": 10,
        "success": "Plan printed in <=5 s, >=3 cm table clearance, grasp within 1 cm of bottle center "
                   "height, joints in limits. Unreachable: NO_PLAN with a reason, no motion.",
        "outcomes": ["SUCCESS", "NO_PLAN", "COLLISION", "TIMEOUT", "ERROR"],
        "metrics": ["plan_time_s", "num_steps", "min_table_clearance_cm", "grasp_height_offset_cm"],
    },
    "T1b": {
        "name": "Execute grasp plan from known coordinates",
        "setup": "Bottle on a taped mark, measured coordinates passed straight to planner -> motion.",
        "conditions": ["left_upright", "center_upright", "right_upright"],
        "trials": 10,
        "success": "No table contact, grips the middle, lifts >=5 cm, holds 5 s, done in <=30 s.",
        "outcomes": ["SUCCESS", "MISS", "COLLISION", "DROP", "TIMEOUT", "ERROR"],
        "metrics": ["exec_time_s", "failed_step", "grip_location"],
    },
    "T2": {
        "name": "Detection accuracy (camera only)",
        "setup": "Base and arm still, head scans. Tape-measure the bottle position.",
        "conditions": ["0.5m_left", "0.5m_center", "0.5m_right", "1.0m_left", "1.0m_center",
                       "1.0m_right", "1.5m_left", "1.5m_center", "1.5m_right"],
        "trials": 10,
        "success": "Detected in <=30 s, position within 3 cm, detection is the bottle (not cap/distractor).",
        "outcomes": ["SUCCESS", "NOT_FOUND", "MISS", "FALSE_POSITIVE", "TIMEOUT", "ERROR"],
        "metrics": ["detect_time_s", "confidence", "error_cm"],
    },
    "T3": {
        "name": "Remember and track the bottle while the base moves",
        "auto": "ros2 run bottle_tracking t3_tracking_tester",
        "setup": "Bottle detected and centered. Part A tracking on, Part B tracking off.",
        "conditions": ["A_rot+30", "A_rot-30", "A_rot+90", "A_rot-90", "A_rot180", "A_fwd0.3", "A_back0.3",
                       "B_rot+30", "B_rot-30", "B_rot+90", "B_rot-90", "B_rot180"],
        "trials": 10,
        "success": "A: in view >=90% of motion, within 10 deg during, within 5 deg 2 s after. "
                   "B: head points back at saved spot in <=3 s, within 5 deg, drift <=3 cm.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "TIMEOUT", "ERROR"],
        "metrics": ["percent_frames_in_view", "max_tracking_error_deg", "reacquire_time_s", "position_drift_cm"],
    },
    "T4": {
        "name": "'Nothing there' recovery when the bottle disappears",
        "setup": "Robot detects the bottle, then a person removes it at the stage below.",
        "conditions": ["removed_before_positioning", "removed_before_grasp", "removed_while_extending"],
        "trials": 10,
        "success": "No grasp at empty spot; if extending, stops and retracts without closing; "
                   "back to Detect in <=10 s; no crash.",
        "outcomes": ["SUCCESS", "MISS", "TIMEOUT", "ERROR"],
        "metrics": ["recovery_time_s", "gripper_closed", "next_state"],
    },
    "T5": {
        "name": "Search with the head first, base only if needed",
        "auto": "ros2 run bottle_tracking t5_search_tester",
        "setup": "Memory cleared before each trial. Tape-measure the bottle position.",
        "conditions": ["in_head_range_left", "in_head_range_right", "edge_of_pan_range",
                       "behind_robot", "far_2_to_3m"],
        "trials": 10,
        "success": "In head range: found with 0 base rotation. Otherwise <=1 full rotation. "
                   "Found in <=60 s, position within 3 cm (5 cm at 2-3 m).",
        "outcomes": ["SUCCESS", "NOT_FOUND", "MISS", "FALSE_POSITIVE", "TIMEOUT", "ERROR"],
        "metrics": ["search_time_s", "base_rotation_deg_during_search", "num_base_rotations", "error_cm"],
    },
    "T6": {
        "name": "Depth / distance estimation (perception only)",
        "setup": "Camera still, bottle straight ahead at a measured distance.",
        "conditions": ["0.4m", "0.6m", "0.8m", "1.0m", "1.5m"],
        "trials": 10,
        "success": "Valid depth, error <=2 cm (<=3 cm at 1.5 m).",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["true_m", "est_m", "error_cm", "signed_error_cm"],
    },
    "T7": {
        "name": "Camera to base transform (TF) accuracy",
        "setup": "Bottle on a measured spot; move the head to each pose.",
        "conditions": ["center", "pan_left", "pan_right", "tilt_down", "tilt_down_pan"],
        "trials": 10,
        "success": "Transformed position within 2 cm of measured.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["head_pan", "head_tilt", "error_cm"],
    },
    "T8": {
        "name": "Grasp only (gripper isolated)",
        "setup": "Place the gripper around the bottle directly; test close -> lift -> hold.",
        "conditions": ["effort_low_middle", "effort_med_middle", "effort_high_middle", "effort_med_near_cap"],
        "trials": 10,
        "success": "Middle of fingers contacts bottle, lifts >=5 cm, holds 5 s, not crushed.",
        "outcomes": ["SUCCESS", "MISS", "DROP", "ERROR"],
        "metrics": ["gripper_effort", "grip_width", "grasp_height_cm"],
    },
    "T9": {
        "name": "Bottle orientation classification",
        "setup": "Log the TRUE orientation as the condition.",
        "conditions": ["upright", "lying_down", "elevated"],
        "trials": 30,
        "success": "Predicted orientation matches true orientation.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "ERROR"],
        "metrics": ["predicted_orientation", "confidence"],
    },
    "T10": {
        "name": "In-hand verification at the head camera",
        "setup": "Part A bottle in gripper, Part B empty gripper.",
        "conditions": ["A_bottle_in_gripper", "B_empty_gripper"],
        "trials": 10,
        "success": "A: confirms pill_bottle in <=5 s. B: verification correctly fails.",
        "outcomes": ["SUCCESS", "MISS", "FALSE_POSITIVE", "TIMEOUT", "ERROR"],
        "metrics": ["verify_time_s", "confidence"],
    },
    "T11": {
        "name": "Retry and failure handling",
        "setup": "Remove the bottle during the grasp, or use a test flag that forces a failed grasp check.",
        "conditions": ["fail_once_then_succeed", "fail_every_time"],
        "trials": 5,
        "success": "Re-detects before each retry, max 3 retries (4 attempts), then reports, stows, stops.",
        "outcomes": ["SUCCESS", "MISS", "TIMEOUT", "ERROR"],
        "metrics": ["attempts", "final_state"],
    },
    "T12": {
        "name": "Bottle moved while the robot is paused",
        "auto": "ros2 run bottle_tracking t12_moved_bottle_tester",
        "setup": "Robot finds the bottle, pauses, you move it (or not), robot resumes.",
        "conditions": ["moved_small_5_15cm", "moved_across_table", "moved_out_of_head_view", "not_moved"],
        "trials": 10,
        "success": "No arm motion to old spot, notices in <=5 s, finds new spot in <=60 s within 3 cm. "
                   "Not moved: keeps saved position, no new search.",
        "outcomes": ["SUCCESS", "MISS", "NOT_FOUND", "FALSE_POSITIVE", "TIMEOUT", "ERROR"],
        "metrics": ["move_distance_cm", "time_to_notice_s", "search_time_s", "base_rotation_deg", "error_cm"],
    },
    "T13": {
        "name": "End-to-end full run",
        "setup": "Full demo, no human help.",
        "conditions": ["pos1_upright", "pos2_upright", "pos3_upright"],
        "trials": 10,
        "success": "Bottle placed back upright within 5 cm of original spot in <=3 min.",
        "outcomes": ["SUCCESS", "NOT_FOUND", "NO_PLAN", "COLLISION", "MISS", "DROP", "TIMEOUT", "ERROR"],
        "metrics": ["total_time_s", "failed_stage", "attempts"],
    },
    "T14": {
        "name": "False positive check (empty scene and distractors)",
        "setup": "Base and arm still, head sweeps. No pill bottle unless the condition says so.",
        "conditions": ["empty_table", "water_bottle", "cup", "phone", "medicine_box", "bottle_plus_distractors"],
        "trials": 10,
        "success": "No pill_bottle detection without a bottle. Mixed scene: picks the bottle within 3 cm.",
        "outcomes": ["SUCCESS", "FALSE_POSITIVE", "NOT_FOUND", "TIMEOUT", "ERROR"],
        "metrics": ["max_false_confidence", "detected_class", "error_cm"],
    },
    "T15a": {
        "name": "Approach planning to a far-away bottle (no motion)",
        "auto": "ros2 run bottle_tracking t15a_approach_plan_tester   (no robot needed)",
        "setup": "Bottle position typed in, planner never moves the robot.",
        "conditions": ["ahead_1.0m", "ahead_2.0m", "ahead_3.0m", "left_1.5m", "right_1.5m",
                       "behind_1.5m", "already_in_reach", "unreachable_far", "unreachable_high"],
        "trials": 10,
        "success": "Plan in <=2 s, bottle predicted in reach and within 2 cm of arm line, >=10 cm "
                   "table clearance, <=2 rotations + 1 drive. Unreachable: NO_PLAN with a reason.",
        "outcomes": ["SUCCESS", "NO_PLAN", "MISS", "COLLISION", "TIMEOUT", "ERROR"],
        "metrics": ["plan_time_s", "total_rotation_deg", "total_drive_m", "predicted_align_error_cm"],
    },
    "T15b": {
        "name": "Approach motion: drive the base to the bottle",
        "auto": "ros2 run bottle_tracking t15b_approach_motion_tester",
        "setup": "Bottle on a taped mark; measured position goes straight to the approach planner.",
        "conditions": ["ahead_1.0m", "ahead_2.0m", "ahead_3.0m", "left_1.5m", "right_1.5m", "behind_1.5m"],
        "trials": 10,
        "success": "No contact, bottle in reach and within 3 cm of arm line, heading within 3 deg, "
                   ">=10 cm from table, in view >=90% of motion, <=45 s.",
        "outcomes": ["SUCCESS", "MISS", "COLLISION", "TIMEOUT", "ERROR"],
        "metrics": ["exec_time_s", "final_align_error_cm", "table_clearance_cm", "failed_step"],
    },
}

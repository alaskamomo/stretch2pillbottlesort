# T9: Integration test: movement + gripper in one run

## What this test does
Runs find → plan → drive → reach → grip → lift → verify at the head camera → put back as one run, and logs
pass/fail and time per stage. On a missed grab it re-detects and retries (up to 3 more times).
Run: `ros2 run pill_bottle_nodes task_manager` (`-p read_label:=false` to leave label reading to T16), then log
each run with `python3 testing/run_manual_test.py T9`.
Conditions: normal, grab fails once, grab fails every time, empty claw (verification must fail).

## Accepted results
- [ ] Full run completes with no collision in 4 of 5 trials
- [ ] Each stage's pass/fail and time is logged
- [ ] Missed grab → re-detects and retries, stops after 4 total attempts, then reports and stows
- [ ] Verification confirms the bottle in ≤ 5 s, and fails correctly with an empty claw

## Possible issues
- One flaky stage fails the whole run, so check the per-stage log to find which
- Grip judged by motor effort; a soft or empty bottle may read as "missed"
- Retries can drift the base further from the bottle each time

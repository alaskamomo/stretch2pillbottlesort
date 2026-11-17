# T8: Label reading: gripper camera vs. head camera (CLOSED → T16)

## What this test does
Compared which camera reads the label better: the black-and-white claw camera or the head camera.

## Accepted results
- [x] Decision: **head camera.** The claw camera was low resolution, black and white, and has been removed.
- [x] Label reading continues in **T16** (wrist turns, lift change, regrasp, Gemini → EasyOCR).

## Possible issues
- If the head camera can't get close enough to read small text, revisit a better claw camera (color, higher resolution).

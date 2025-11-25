# Pi0.5 OpenPI Integration - Implementation Notes

## Overview

Successfully integrated Pi0.5 ALOHA policy with robosuite's TwoArmLift environment. The implementation follows the openpi LIBERO example pattern and connects via WebSocket.

## Changes Made

### 1. Rewrote `test_pick_place.py`

- **Old**: HTTP-based `BimanualPolicyClient` using EEF pose observations
- **New**: WebSocket-based `openpi_client.WebsocketClientPolicy` using joint-space observations

Key differences:
- Protocol: HTTP → WebSocket (with msgpack serialization)
- State: EEF pose (Cartesian) → Joint positions  
- Action chunking: Single actions → Chunked execution with replanning

### 2. Observation Format Conversion

Implemented `make_aloha_obs()` function that converts robosuite observations to openpi ALOHA format:

```python
Input (robosuite):
- robot0_joint_pos: (12,) joint positions
- robot0_left_gripper_qpos: left gripper state
- robot0_right_gripper_qpos: right gripper state
- Images: HWC uint8

Output (openpi ALOHA):
{
    "state": (14,) [left_arm(6), left_grip(1), right_arm(6), right_grip(1)],
    "images": {
        "cam_high": (3, 224, 224) CHW uint8,
        "cam_left_wrist": (3, 224, 224),
        "cam_right_wrist": (3, 224, 224),
    },
    "prompt": str
}
```

### 3. Critical: Joint Ordering Fix

**Issue**: Robosuite bimanual robots (Baxter, Tiago) use `[right_arm, left_arm]` ordering, but openpi ALOHA expects `[left_arm, right_arm]`.

**Solution**: State mapping in `make_aloha_obs()` SWAPS the joint order:
```python
state = np.concatenate([
    joint_pos[6:12],    # left arm (from robosuite's right=0:6, left=6:12)
    left_gripper[:1],   
    joint_pos[:6],      # right arm
    right_gripper[:1],
])
```

**Verification Needed**: If the robot exhibits mirrored behavior (e.g., left arm moves when right should move), the joint ordering assumption may be incorrect and needs adjustment.

## Setup Instructions

### Server Side (GPU Machine)

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py --env ALOHA --port 8000
```

This loads the Pi0.5 ALOHA checkpoint (`gs://openpi-assets/checkpoints/pi05_base`).

### Client Side (Simulation Machine)

1. Install openpi_client:
```bash
cd /path/to/openpi/packages/openpi-client
pip install -e .
# or manually: pip install websockets msgpack numpy
```

2. Run evaluation:
```bash
cd /path/to/robosuite/tcr_eval
python test_pick_place.py --policy-host <server_ip> --policy-port 8000
```

## Files Modified/Created

- ✅ `tcr_eval/test_pick_place.py` - Rewritten to use openpi WebSocket client
- ⚠️ `tcr_eval/policy_client.py` - No longer needed (HTTP-based, kept for reference)
- ⚠️ `tcr_eval/policy_server.py` - No longer needed (HTTP-based, kept for reference)
- ✅ `tcr_eval/INTEGRATION_NOTES.md` - This file

## Known Limitations

1. **Domain Gap**: Pi0.5 ALOHA was trained on real Trossen ALOHA hardware data, NOT robosuite simulation. Performance may be poor without fine-tuning.

2. **Joint Ordering**: The joint order swap ([right,left] → [left,right]) is based on inspection of other robosuite bimanual robots (Baxter, Tiago). The Aloha robot definition is in the external `robosuite_models` package and wasn't directly verified.

3. **Gripper Normalization**: openpi applies ALOHA-specific gripper transformations internally (`adapt_to_pi=True` in `aloha_policy.py`). These may not match robosuite's gripper range.

## Testing Checklist

When first running the integration:

- [ ] Server connects successfully (check logs)
- [ ] First inference completes without errors
- [ ] Action shapes are correct: (chunk_size, 14)
- [ ] Robot moves in expected directions (not mirrored)
- [ ] Grippers open/close correctly
- [ ] No systematic left-right swapping observed

If behavior is mirrored, swap the joint order in `make_aloha_obs()`.

## References

- OpenPI LIBERO example: `/path/to/openpi/examples/libero/main.py`
- OpenPI ALOHA format: `/path/to/openpi/src/openpi/policies/aloha_policy.py`
- OpenPI serve script: `/path/to/openpi/scripts/serve_policy.py`


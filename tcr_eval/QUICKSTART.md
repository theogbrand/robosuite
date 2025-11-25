# Quick Start: Pi0.5 with Robosuite

## Prerequisites

1. **GPU Machine** (for Pi0.5 inference):
   - openpi repository installed
   - Access to `gs://openpi-assets/checkpoints/pi05_base`

2. **Simulation Machine** (for robosuite):
   - robosuite with Aloha robot support
   - openpi-client package

## Step 1: Install openpi-client (Simulation Machine)

```bash
# Option 1: Install from openpi repo
cd /path/to/openpi/packages/openpi-client
pip install -e .

# Option 2: Install minimal dependencies
pip install websockets msgpack numpy
```

## Step 2: Start Pi0.5 Server (GPU Machine)

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py --env ALOHA --port 8000
```

Expected output:
```
INFO:openpi.policies.policy:Loading policy from gs://openpi-assets/checkpoints/pi05_base
INFO:__main__:Creating server (host: <hostname>, ip: <ip>)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

## Step 3: Run Evaluation (Simulation Machine)

```bash
cd /path/to/robosuite/tcr_eval
python test_pick_place.py --policy-host <gpu_server_ip> --policy-port 8000 --n-episodes 10
```

For localhost testing (server and sim on same machine):
```bash
python test_pick_place.py --n-episodes 10
```

## Step 4: Verify Behavior

Watch the robot behavior. If movements are mirrored (left/right swapped):

1. Open `test_pick_place.py`
2. In `make_aloha_obs()`, swap the joint indices:

```python
# Current (assumes robosuite uses [right, left] ordering):
state = np.concatenate([
    joint_pos[6:12],    # left from robosuite's left=6:12
    left_gripper[:1],
    joint_pos[:6],      # right from robosuite's right=0:6
    right_gripper[:1],
])

# If mirrored, try [left, right] ordering instead:
state = np.concatenate([
    joint_pos[:6],      # left from robosuite's left=0:6
    left_gripper[:1],
    joint_pos[6:12],    # right from robosuite's right=6:12
    right_gripper[:1],
])
```

## Command Line Options

```bash
python test_pick_place.py \
    --n-episodes 20 \              # Number of evaluation episodes
    --horizon 500 \                # Max steps per episode
    --policy-host localhost \      # Pi0.5 server hostname
    --policy-port 8000 \           # Pi0.5 server port
    --task-description "Lift the pot using both arms" \
    --replan-steps 5 \             # Actions before re-inference
    --no-display                   # Disable OpenCV display
```

## Testing Without Policy

To test robosuite environment without Pi0.5:

```bash
python test_pick_place.py --no-policy --n-episodes 5
```

This uses random actions to verify the environment setup.

## Troubleshooting

### Connection Refused
- Check server is running: `netstat -an | grep 8000`
- Check firewall allows port 8000
- Verify IP address is correct

### Import Error: openpi_client
```bash
cd /path/to/openpi/packages/openpi-client && pip install -e .
```

### Robot Not Moving / Random Actions
- Check server logs for errors
- Verify `obs_dict` shapes in first inference (see logging output)
- Action shape should be `(chunk_size, 14)`

### Mirrored Behavior
- See Step 4 above to swap joint ordering

### Poor Performance
- Expected! Pi0.5 trained on real hardware, not robosuite sim
- Consider fine-tuning on robosuite data
- Or try `--env ALOHA_SIM` (Pi0 trained on gym_aloha sim)

## Expected Performance

⚠️ **Important**: Pi0.5 ALOHA checkpoint was trained on:
- Real Trossen ALOHA robot hardware
- Real-world manipulation tasks
- Different visual domain, dynamics, gripper behavior

**Expected baseline performance in robosuite**: 
- Random or exploratory behavior (0-5% success)
- May require significant fine-tuning to achieve good performance

To get better results:
1. Collect robosuite demonstrations
2. Fine-tune Pi0.5 on robosuite data
3. Or try training from scratch with Pi0 architecture on robosuite


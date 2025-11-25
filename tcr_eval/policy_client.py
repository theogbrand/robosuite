"""
Bimanual Policy Client for VLA Policy Server Integration

This module provides an HTTP client to connect robosuite's bimanual robot simulation
to a VLA (Vision-Language-Action) policy server. It handles observation preprocessing,
HTTP communication, and action format conversion.

Server Contract:
    POST /act
    Content-Type: application/json
    Request: {"observation": {...}}
    Response: {"action.right_arm": [...], "action.left_arm": [...]}
"""

import math
from typing import Optional

import numpy as np

try:
    import json_numpy
    json_numpy.patch()
    import requests
except ImportError as e:
    raise ImportError(
        "Missing dependencies for policy client. Install with:\n"
        "  pip install requests json-numpy"
    ) from e


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to axis-angle representation.
    
    Args:
        quat: (4,) quaternion in xyzw format
        
    Returns:
        (3,) axis-angle exponential coordinates
    """
    # Clip w component to valid range
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    
    if math.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(w)) / den


class BimanualPolicyClient:
    """
    HTTP client for bimanual VLA policy server.
    
    Sends observations (3 camera images + robot state) and receives
    coordinated actions for both arms in a single call.
    
    Args:
        host: Server hostname (default: "localhost")
        port: Server port (default: 8000)
        timeout: Request timeout in seconds (default: 30)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: float = 30.0,
    ):
        self.url = f"http://{host}:{port}/act"
        self.timeout = timeout
        self._session = requests.Session()
    
    def _process_observation(
        self,
        obs: dict,
        task_description: str,
    ) -> dict:
        """
        Convert robosuite observation dict to policy server format.
        
        Args:
            obs: Raw observation dict from robosuite env
            task_description: Language instruction for the task
            
        Returns:
            Formatted observation dict for policy server
        """
        # Extract images (robosuite returns HxWx3 uint8)
        agentview = obs["agentview_image"]
        wrist_right = obs["robot0_wrist_cam_right_image"]
        wrist_left = obs["robot0_wrist_cam_left_image"]
        
        # Extract robot state for right arm
        right_eef_pos = obs["robot0_right_eef_pos"]  # (3,)
        right_eef_quat = obs["robot0_right_eef_quat"]  # (4,) xyzw
        right_eef_rpy = quat2axisangle(right_eef_quat)  # (3,) axis-angle
        right_gripper = obs["robot0_right_gripper_qpos"]  # (n,)
        
        # Extract robot state for left arm
        left_eef_pos = obs["robot0_left_eef_pos"]  # (3,)
        left_eef_quat = obs["robot0_left_eef_quat"]  # (4,) xyzw
        left_eef_rpy = quat2axisangle(left_eef_quat)  # (3,) axis-angle
        left_gripper = obs["robot0_left_gripper_qpos"]  # (n,)
        
        # Construct state vectors: [x, y, z, roll, pitch, yaw, gripper...]
        right_state = np.concatenate([right_eef_pos, right_eef_rpy, right_gripper])
        left_state = np.concatenate([left_eef_pos, left_eef_rpy, left_gripper])
        
        # Format for policy server (add batch dimension)
        return {
            "video.agentview": np.expand_dims(agentview, axis=0),  # (1, H, W, 3)
            "video.wrist_right": np.expand_dims(wrist_right, axis=0),  # (1, H, W, 3)
            "video.wrist_left": np.expand_dims(wrist_left, axis=0),  # (1, H, W, 3)
            "state.right_arm": np.expand_dims(right_state, axis=0),  # (1, 7+)
            "state.left_arm": np.expand_dims(left_state, axis=0),  # (1, 7+)
            "annotation.human.action.task_description": [task_description],
        }
    
    def _convert_to_robosuite_action(
        self,
        action_response: dict,
        action_idx: int = 0,
    ) -> np.ndarray:
        """
        Convert policy server response to robosuite action format.
        
        Args:
            action_response: Dict with 'action.right_arm' and 'action.left_arm'
            action_idx: Index of action to extract from chunk (default: 0)
            
        Returns:
            (14,) action array: [right_arm(7), left_arm(7)]
            Each arm: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        right_action = np.atleast_2d(action_response["action.right_arm"])[action_idx]
        left_action = np.atleast_2d(action_response["action.left_arm"])[action_idx]
        
        # Robosuite expects concatenated actions: [right, left]
        return np.concatenate([right_action, left_action]).astype(np.float32)
    
    def get_action(
        self,
        obs: dict,
        task_description: str = "Complete the task",
        action_idx: int = 0,
    ) -> np.ndarray:
        """
        Query policy server for bimanual action.
        
        Args:
            obs: Raw observation dict from robosuite env
            task_description: Language instruction for the task
            action_idx: Which action to use from returned chunk (default: 0)
            
        Returns:
            (14,) action array for robosuite: [right_arm(7), left_arm(7)]
            
        Raises:
            requests.RequestException: On network/server errors
            KeyError: If response missing expected action keys
        """
        # Preprocess observation
        processed_obs = self._process_observation(obs, task_description)
        
        # Send to policy server
        response = self._session.post(
            self.url,
            json={"observation": processed_obs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        # Convert response to robosuite action
        action_response = response.json()
        return self._convert_to_robosuite_action(action_response, action_idx)
    
    def ping(self) -> bool:
        """Check if server is reachable."""
        try:
            ping_url = self.url.replace("/act", "/ping")
            response = self._session.get(ping_url, timeout=5.0)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def close(self):
        """Close the HTTP session."""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


"""
Bimanual Policy Server for VLA Models (Pi0, GR00T, etc.)

This module provides an HTTP server that serves VLA (Vision-Language-Action) models
for bimanual robot control. It's compatible with the BimanualPolicyClient.

Server Contract:
    POST /act - Get action prediction from observation
    GET  /ping - Health check

Example Usage:
    # Start server with dummy policy (for testing)
    python policy_server.py --port 8000

    # With custom policy (user implements Pi0Policy subclass)
    python policy_server.py --port 8000 --policy-class my_policies:Pi0Policy
"""

import argparse
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import json_numpy
    json_numpy.patch()
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    raise ImportError(
        "Missing dependencies for policy server. Install with:\n"
        "  pip install fastapi uvicorn json-numpy"
    ) from e


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BimanualPolicy(ABC):
    """
    Abstract base class for bimanual robot policies.
    
    Users should subclass this to implement their specific model (Pi0, GR00T, etc.).
    """
    
    @abstractmethod
    def predict(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Predict bimanual actions from observations.
        
        Args:
            obs: Observation dictionary with keys:
                - video.agentview: (1, H, W, 3) uint8
                - video.wrist_right: (1, H, W, 3) uint8
                - video.wrist_left: (1, H, W, 3) uint8
                - state.right_arm: (1, D) float
                - state.left_arm: (1, D) float
                - annotation.human.action.task_description: [str]
        
        Returns:
            Dictionary with:
                - action.right_arm: (chunk_size, 7) or (7,)
                - action.left_arm: (chunk_size, 7) or (7,)
        """
        raise NotImplementedError


class DummyPolicy(BimanualPolicy):
    """
    Dummy policy that returns zero actions.
    Useful for testing server integration without a real model.
    """
    
    def __init__(self, action_chunk_size: int = 1):
        self.action_chunk_size = action_chunk_size
        logger.info(f"Initialized DummyPolicy (returns zeros, chunk_size={action_chunk_size})")
    
    def predict(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return zero actions for both arms."""
        if self.action_chunk_size == 1:
            return {
                "action.right_arm": np.zeros(7, dtype=np.float32),
                "action.left_arm": np.zeros(7, dtype=np.float32),
            }
        else:
            return {
                "action.right_arm": np.zeros((self.action_chunk_size, 7), dtype=np.float32),
                "action.left_arm": np.zeros((self.action_chunk_size, 7), dtype=np.float32),
            }


class PolicyServer:
    """HTTP server for bimanual policy inference."""
    
    def __init__(
        self,
        policy: BimanualPolicy,
        port: int = 8000,
        host: str = "0.0.0.0",
    ):
        self.policy = policy
        self.port = port
        self.host = host
        self.app = FastAPI(title="Bimanual Policy Server", version="1.0.0")
        
        # Register endpoints
        self.app.post("/act")(self.act)
        self.app.get("/ping")(self.ping)
        self.app.get("/health")(self.health)
    
    def preprocess_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation for model input.
        
        Converts images from (1, H, W, 3) uint8 to (1, 3, H, W) float32 [0, 1].
        Other modalities pass through as-is.
        """
        processed = {}
        
        for key, value in obs.items():
            if key.startswith("video."):
                # Convert images: (B, H, W, C) uint8 -> (B, C, H, W) float32
                arr = np.array(value)
                if arr.dtype == np.uint8:
                    # Normalize to [0, 1]
                    arr = arr.astype(np.float32) / 255.0
                # Transpose to channel-first
                if len(arr.shape) == 4:  # (B, H, W, C)
                    arr = np.transpose(arr, (0, 3, 1, 2))  # -> (B, C, H, W)
                processed[key] = arr
            else:
                # Pass through state, language annotations
                processed[key] = value
        
        return processed
    
    async def act(self, payload: Dict[str, Any]) -> JSONResponse:
        """
        Action prediction endpoint.
        
        Expects: {"observation": {...}}
        Returns: {"action.right_arm": [...], "action.left_arm": [...]}
        """
        try:
            if "observation" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'observation' field in payload"
                )
            
            obs = payload["observation"]
            
            # Preprocess observation
            processed_obs = self.preprocess_observation(obs)
            
            # Run model inference
            action = self.policy.predict(processed_obs)
            
            # Validate output
            if "action.right_arm" not in action or "action.left_arm" not in action:
                raise ValueError(
                    "Policy must return 'action.right_arm' and 'action.left_arm'"
                )
            
            # Return as JSON (json-numpy handles numpy arrays)
            return JSONResponse(content=action)
        
        except Exception as e:
            logger.error(f"Error in /act endpoint: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    async def ping(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}
    
    async def health(self) -> Dict[str, str]:
        """Health check endpoint (alias for /ping)."""
        return {"status": "healthy", "model": "BimanualPolicy"}
    
    def run(self):
        """Start the HTTP server."""
        logger.info(f"Starting Bimanual Policy Server on {self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  POST /act    - Get action prediction")
        logger.info("  GET  /ping   - Health check")
        logger.info("  GET  /health - Health check (alias)")
        uvicorn.run(self.app, host=self.host, port=self.port)


def load_policy_class(policy_class_str: Optional[str]) -> type:
    """
    Dynamically load a policy class from module:ClassName string.
    
    Example: "my_policies:Pi0Policy"
    """
    if policy_class_str is None:
        return DummyPolicy
    
    import importlib
    import os
    import sys
    from pathlib import Path
    
    # Add current directory to path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        module_path, class_name = policy_class_str.split(":", 1)
        logger.info(f"Loading policy class: {module_path}.{class_name}")
        
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'"
            )
        
        policy_class = getattr(module, class_name)
        
        # Verify it's a BimanualPolicy subclass
        if not issubclass(policy_class, BimanualPolicy):
            raise TypeError(
                f"{class_name} must inherit from BimanualPolicy"
            )
        
        return policy_class
    
    except Exception as e:
        logger.error(f"Failed to load policy class: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bimanual Policy Server for VLA Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with dummy policy (testing)
  python policy_server.py --port 8000
  
  # Start with custom Pi0 policy
  python policy_server.py --port 8000 --policy-class my_policies:Pi0Policy --model-path ./model.safetensors
  
  # With GPU
  python policy_server.py --device cuda:0 --policy-class my_policies:Pi0Policy
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for model inference (default: cuda:0 if available, else cpu)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (e.g., model.safetensors)"
    )
    
    parser.add_argument(
        "--policy-class",
        type=str,
        default=None,
        help="Python import path to policy class (format: module:ClassName, e.g., my_policies:Pi0Policy)"
    )
    
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=1,
        help="Action chunk size for dummy policy (default: 1)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load policy class
    policy_class = load_policy_class(args.policy_class)
    
    # Instantiate policy
    if policy_class == DummyPolicy:
        logger.warning("Using DummyPolicy - this returns zero actions!")
        policy = DummyPolicy(action_chunk_size=args.action_chunk_size)
    else:
        # User-provided policy class
        # Assume it takes model_path and device as kwargs
        policy_kwargs = {}
        if args.model_path:
            policy_kwargs["model_path"] = args.model_path
        if hasattr(policy_class.__init__, "__code__"):
            param_names = policy_class.__init__.__code__.co_varnames
            if "device" in param_names:
                policy_kwargs["device"] = args.device
        
        logger.info(f"Instantiating {policy_class.__name__} with kwargs: {policy_kwargs}")
        policy = policy_class(**policy_kwargs)
    
    # Create and run server
    server = PolicyServer(policy, port=args.port, host=args.host)
    server.run()


if __name__ == "__main__":
    main()


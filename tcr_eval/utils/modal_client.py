"""
Modal policy client for connecting to Modal-deployed OpenPI inference server.

This is a simplified version that accepts a direct WebSocket URL instead of
discovering it via Modal SDK. It handles the Modal-specific initialization protocol:
1. Connect and receive empty metadata from server
2. Send initialization message with model configuration
3. Receive policy metadata
4. Ready for inference
"""

import logging
import websockets
import websockets.sync.client
from typing import Any


import utils.msgpack_numpy as msgpack_numpy



logger = logging.getLogger(__name__)


class ModalClientPolicy:
    """Client for connecting to Modal serverless policy endpoint.
    
    This class handles the Modal-specific initialization protocol which differs
    from the standard WebSocket policy server in that it dynamically loads models
    based on the initialization message sent by the client.
    """
    
    def __init__(
        self,
        url: str,
        hf_repo_id: str,
        folder_path: str,
        config_name: str,
        prompt: str | None = None,
        dataset_repo_id: str | None = None,
        stats_json_path: str | None = None,
        connect_timeout: float | None = None,
    ):
        """Initialize and connect to Modal endpoint.
        
        Args:
            url: WebSocket URL (wss://...) of the Modal endpoint
            hf_repo_id: HuggingFace repo ID for the model
            folder_path: Path to checkpoint folder within the repo
            config_name: Config name to use for loading the policy
            prompt: Optional default prompt for the policy
            dataset_repo_id: Optional HuggingFace dataset repo for norm_stats
            stats_json_path: Path to stats.json within the dataset repo
            connect_timeout: Optional connection timeout in seconds
        """
        self._packer = msgpack_numpy.Packer()
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._policy_metadata: dict = {}
        self._url = url

        # Store model configuration
        self._hf_repo_id = hf_repo_id
        self._folder_path = folder_path
        self._config_name = config_name
        self._prompt = prompt
        self._dataset_repo_id = dataset_repo_id
        self._stats_json_path = stats_json_path
        self._connect_timeout = connect_timeout
        
        # Initialize connection and load policy
        logger.info(f"Connecting to Modal endpoint at {self._url}...")
        self._connect_and_initialize(connect_timeout)
        logger.info(f"✓ Connected and ready for inference")
    
    def _connect_and_initialize(self, timeout: float | None = None):
        """Connect to server and complete initialization protocol."""
        # Connect to WebSocket
        self._ws = websockets.sync.client.connect(
            self._url,
            compression=None,
            max_size=None,
            open_timeout=timeout,
            close_timeout=timeout,
            ping_interval=20,   # Keep Modal load balancer alive
            ping_timeout=30,
        )
        
        # PHASE 1: Receive initial empty metadata from server
        logger.info("Waiting for initial server metadata...")
        init_metadata = msgpack_numpy.unpackb(self._ws.recv())
        logger.info(f"Received initial metadata: {init_metadata}")
        
        # PHASE 2: Send initialization message with model configuration
        init_message = {
            "hf_repo_id": self._hf_repo_id,
            "folder_path": self._folder_path,
            "config_name": self._config_name,
        }
        if self._prompt:
            init_message["prompt"] = self._prompt
        if self._dataset_repo_id:
            init_message["dataset_repo_id"] = self._dataset_repo_id
        if self._stats_json_path:
            init_message["stats_json_path"] = self._stats_json_path
        
        logger.info(f"Sending model configuration: {self._config_name} from {self._hf_repo_id}/{self._folder_path}")
        self._ws.send(self._packer.pack(init_message))
        
        # PHASE 3: Receive policy metadata
        logger.info("Waiting for policy to load (this may take a while for cold starts)...")
        policy_metadata_data = self._ws.recv()
        
        # Check if we got an error message (text traceback)
        if isinstance(policy_metadata_data, str):
            error_msg = f"Error from server during policy loading:\n{policy_metadata_data}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self._policy_metadata = msgpack_numpy.unpackb(policy_metadata_data)
        logger.info(f"Received policy metadata: {self._policy_metadata}")
    
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run inference with automatic reconnection on connection loss.
        
        Args:
            obs: Observation dictionary with keys like 'state', 'images', etc.
            
        Returns:
            Action dictionary with keys like 'actions', 'server_timing', etc.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._ws is None:
                    raise RuntimeError("Not connected to server")
                
                data = self._packer.pack(obs)
                self._ws.send(data)
                response = self._ws.recv()
                
                if isinstance(response, str):
                    error_msg = f"Error in inference server:\n{response}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                return msgpack_numpy.unpackb(response)
                
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                logger.warning(f"Connection lost (attempt {attempt+1}/{max_retries}): {e}")
                self._ws = None
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to reconnect after {max_retries} attempts") from e
                logger.info("Reconnecting to Modal endpoint...")
                self._connect_and_initialize(self._connect_timeout)
    
    def get_policy_metadata(self) -> dict:
        """Get the policy metadata received during initialization."""
        return self._policy_metadata
    
    def close(self):
        """Close the WebSocket connection."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None
            logger.info("Connection closed")


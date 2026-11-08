"""Isaac Lab VecEnv wrapper for SB3 compatibility (work package W7.x).

Provides an SB3-compatible VecEnv interface around Isaac Lab's ManagerBasedRLEnv.
"""

from __future__ import annotations

import numpy as np
from typing import Any
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces


class IsaacLabVecEnvWrapper(VecEnv):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return False
    def env_method(self, method_name, *args, indices=None, **kwargs):
        return []
    def get_attr(self, attr, indices=None):
        return []
    def set_attr(self, attr, value, indices=None):
        pass
    """Wrap an Isaac Lab ManagerBasedRLEnv for SB3 PPO compatibility.

    SB3's PPO expects:
    - reset() -> (n_envs, obs_dim) np.ndarray
    - step(actions) -> (obs, rewards, dones, infos) with 4-tuple
    - num_envs, observation_space, action_space properties
    - close() method

    Isaac Lab's ManagerBasedRLEnv uses:
    - reset() -> (obs_dict, info_dict)
    - step(actions) -> (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
    - Tensor-based observations (torch.Tensor on device)

    This wrapper handles the conversion and provides the expected numpy arrays.
    """

    def __init__(
        self,
        env: Any,
        num_envs: int,
        seed: int = 0,
    ):
        """Initialize the wrapper.

        Args:
            env: Isaac Lab ManagerBasedRLEnv instance.
            num_envs: Number of parallel environments.
            seed: Base seed for reproducibility.
        """
        self._env = env
        self._num_envs = num_envs
        self._seed = seed
        self._closed = False

        # Isaac Lab uses torch tensors; we need to know the device
        self._device = getattr(env, "device", "cuda")

        # Get observation and action spaces from the environment
        # Isaac Lab uses dictionary spaces; we need to extract the policy observation
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Extract and configure observation/action spaces from Isaac Lab env."""
        # Get the policy observation space (the one used for training)
        # Isaac Lab ManagerBasedRLEnv has observation_space as a Dict space
        # with "policy" key containing the actual observation
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        if hasattr(obs_space, "spaces") and "policy" in obs_space.spaces:
            policy_obs_space = obs_space.spaces["policy"]
        else:
            # Fallback: assume flat observation space
            policy_obs_space = obs_space

        if hasattr(act_space, "spaces") and "policy" in act_space.spaces:
            policy_act_space = act_space.spaces["policy"]
        else:
            policy_act_space = act_space

        # Convert to flat Box spaces for SB3
        if isinstance(policy_obs_space, spaces.Box):
            self.observation_space = policy_obs_space
        elif isinstance(policy_obs_space, spaces.Dict):
            # Flatten dict observation
            total_dim = sum(
                np.prod(s.shape) for s in policy_obs_space.spaces.values()
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32,
            )
        else:
            # Fallback
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(45,),
                dtype=np.float32,
            )

        if isinstance(policy_act_space, spaces.Box):
            self.action_space = policy_act_space
        elif isinstance(policy_act_space, spaces.Dict):
            total_dim = sum(
                np.prod(s.shape) for s in policy_act_space.spaces.values()
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(total_dim,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32,
            )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self) -> np.ndarray:
        """Reset all environments and return observations as numpy array."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")

        # Isaac Lab reset returns (obs_dict, info_dict)
        obs_dict, _ = self._env.reset()

        # Extract policy observation and convert to numpy
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict

        # Convert torch tensor to numpy
        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()
        elif hasattr(obs, "numpy"):
            obs = obs.numpy()

        # Ensure correct shape and dtype
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Queue actions for step_wait (SB3 VecEnv interface)."""
        self._actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Execute the queued actions and return (obs, rewards, dones, infos)."""
        if self._closed:
            raise RuntimeError("Cannot step a closed environment")

        # Convert actions to torch tensor on the correct device
        import torch
        actions_tensor = torch.from_numpy(self._actions).to(self._device)

        # Isaac Lab step returns (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self._env.step(actions_tensor)

        # Extract policy observations
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict

        # Convert to numpy
        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()
        elif hasattr(obs, "numpy"):
            obs = obs.numpy()

        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Extract rewards - Isaac Lab returns dict with "policy" key or flat tensor
        if isinstance(reward_dict, dict) and "policy" in reward_dict:
            rewards = reward_dict["policy"]
        else:
            rewards = reward_dict

        if hasattr(rewards, "cpu"):
            rewards = rewards.cpu().numpy()
        elif hasattr(rewards, "numpy"):
            rewards = rewards.numpy()

        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.ndim == 0:
            rewards = np.array([rewards])

        # Combine terminated and truncated into dones (SB3 style)
        if isinstance(terminated_dict, dict) and "policy" in terminated_dict:
            terminated = terminated_dict["policy"]
        else:
            terminated = terminated_dict

        if isinstance(truncated_dict, dict) and "policy" in truncated_dict:
            truncated = truncated_dict["policy"]
        else:
            truncated = truncated_dict

        if hasattr(terminated, "cpu"):
            terminated = terminated.cpu().numpy()
        elif hasattr(terminated, "numpy"):
            terminated = terminated.numpy()

        if hasattr(truncated, "cpu"):
            truncated = truncated.cpu().numpy()
        elif hasattr(truncated, "numpy"):
            truncated = truncated.numpy()

        dones = np.logical_or(
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool)
        )

        # Build infos list
        infos = []
        for i in range(self._num_envs):
            info = {}
            if isinstance(info_dict, dict):
                # Extract per-environment info
                for k, v in info_dict.items():
                    if hasattr(v, "__len__") and len(v) == self._num_envs:
                        info[k] = v[i]
                    else:
                        info[k] = v
            infos.append(info)

        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Convenience method: step_async + step_wait."""
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        """Close the underlying environment."""
        if not self._closed:
            self._env.close()
            self._closed = True

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the environment (delegates to Isaac Lab)."""
        if hasattr(self._env, "render"):
            return self._env.render(mode=mode)
        return None

    def seed(self, seed: int | None = None) -> list[int] | None:
        """Set the seed for the environment."""
        if seed is not None:
            self._seed = seed
        # Isaac Lab uses its own seeding mechanism
        if hasattr(self._env, "seed"):
            return self._env.seed(seed)
        return [self._seed + i for i in range(self._num_envs)]

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self._env, name)
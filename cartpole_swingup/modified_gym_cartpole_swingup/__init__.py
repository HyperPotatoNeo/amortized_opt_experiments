# pylint:disable=missing-module-docstring
import importlib

from gym.envs.registration import register

register(
    id="ModifiedCartPoleSwingUp-v0",
    entry_point="modified_gym_cartpole_swingup.envs.cartpole_swingup:CartPoleSwingUpV0",
    max_episode_steps=500,
)

register(
    id="ModifiedCartPoleSwingUp-v1",
    entry_point="modified_gym_cartpole_swingup.envs.cartpole_swingup:CartPoleSwingUpV1",
    max_episode_steps=500,
)


if importlib.util.find_spec("torch"):
    register(
        id="ModifiedTorchCartPoleSwingUp-v0",
        entry_point="modified_gym_cartpole_swingup.envs.torch_cartpole_swingup:"
        "TorchCartPoleSwingUpV0",
    )

    register(
        id="ModifiedTorchCartPoleSwingUp-v1",
        entry_point="modified_gym_cartpole_swingup.envs.torch_cartpole_swingup:"
        "TorchCartPoleSwingUpV1",
        max_episode_steps=500,
    )

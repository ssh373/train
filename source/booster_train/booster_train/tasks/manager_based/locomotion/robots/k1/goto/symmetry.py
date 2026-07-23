"""Runtime-resolved K1 left/right reflection for RSL-RL symmetry loss."""

from __future__ import annotations

import torch


def _unwrap_env(env):
    """Resolve Isaac Lab's environment behind one or more RL/Gym wrappers."""
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if hasattr(env, "scene"):
            return env
        if hasattr(env, "unwrapped"):
            candidate = env.unwrapped
        elif hasattr(env, "env"):
            candidate = env.env
        else:
            break
        if candidate is env:
            break
        env = candidate
    raise AttributeError("Unable to resolve an Isaac Lab environment with a 'scene' attribute")


def _joint_permutation(env, device):
    env = _unwrap_env(env)
    names = env.scene["robot"].data.joint_names
    index = {name: i for i, name in enumerate(names)}
    selected = [i for i, name in enumerate(names) if any(x in name for x in ("Hip_", "Knee_", "Ankle_"))]
    selected_names = [names[i] for i in selected]
    local = {name: i for i, name in enumerate(selected_names)}
    permutation, signs = [], []
    for name in selected_names:
        other = name.replace("Left_", "@_").replace("Right_", "Left_").replace("@_", "Right_")
        if other not in index or other not in local:
            raise ValueError(f"K1 mirror counterpart missing for joint {name!r}: expected {other!r}")
        permutation.append(local[other])
        signs.append(-1.0 if ("_Roll" in name or "_Yaw" in name) else 1.0)
    return torch.tensor(permutation, device=device), torch.tensor(signs, device=device)


def _mirror_joints(values, permutation, signs):
    return values[..., permutation] * signs


def mirror_observations(obs, env, obs_type):
    permutation, signs = _joint_permutation(env, obs.device)
    n = len(permutation)
    out = obs.clone()
    # Policy layout: angular velocity, projected gravity, q, qdot, previous action, goal.
    out[..., 0:3] *= torch.tensor([-1.0, 1.0, -1.0], device=obs.device)
    out[..., 3:6] *= torch.tensor([1.0, -1.0, 1.0], device=obs.device)
    offset = 6
    for _ in range(3):
        out[..., offset:offset+n] = _mirror_joints(obs[..., offset:offset+n], permutation, signs)
        offset += n
    out[..., offset:offset+4] *= torch.tensor([1.0, -1.0, -1.0, 1.0], device=obs.device)
    offset += 4
    if obs_type == "critic":
        out[..., offset:offset+3] *= torch.tensor([1.0, -1.0, 1.0], device=obs.device)
        offset += 3 + 1 + 1  # base linear velocity, height, both-feet-grounded
        foot_vel = obs[..., offset:offset+6].reshape(*obs.shape[:-1], 2, 3)
        out[..., offset:offset+6] = foot_vel.flip(-2).mul(
            torch.tensor([1.0, -1.0, 1.0], device=obs.device)).reshape(*obs.shape[:-1], 6)
        offset += 6
        out[..., offset:offset+n] = _mirror_joints(obs[..., offset:offset+n], permutation, signs)
    return out


def mirror_actions(actions, env):
    permutation, signs = _joint_permutation(env, actions.device)
    return _mirror_joints(actions, permutation, signs)


def data_augmentation_func(obs, actions, env, obs_type):
    """Return original+mirrored batches as required by RSL-RL 2.3.1 PPO."""
    augmented_obs = None if obs is None else torch.cat((obs, mirror_observations(obs, env, obs_type)), dim=0)
    augmented_actions = None if actions is None else torch.cat((actions, mirror_actions(actions, env)), dim=0)
    return augmented_obs, augmented_actions

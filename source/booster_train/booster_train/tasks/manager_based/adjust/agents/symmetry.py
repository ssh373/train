"""Left/right reflection for the K1 adjust actor, critic, and actions."""

from __future__ import annotations

import torch


def _unwrap_env(env):
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if hasattr(env, "scene"):
            return env
        candidate = env.unwrapped if hasattr(env, "unwrapped") else getattr(env, "env", env)
        if candidate is env:
            break
        env = candidate
    raise AttributeError("Unable to resolve an Isaac Lab environment")


def _joint_mirror(env, device):
    env = _unwrap_env(env)
    names = env.scene["robot"].data.joint_names
    selected = [
        name for name in names if any(part in name for part in ("Hip_", "Knee_", "Ankle_"))
    ]
    local = {name: i for i, name in enumerate(selected)}
    permutation, signs = [], []
    for name in selected:
        other = name.replace("Left_", "@_").replace("Right_", "Left_").replace("@_", "Right_")
        if other not in local:
            raise ValueError(f"K1 mirror counterpart missing for {name!r}: expected {other!r}")
        permutation.append(local[other])
        signs.append(-1.0 if ("_Roll" in name or "_Yaw" in name) else 1.0)
    return torch.tensor(permutation, device=device), torch.tensor(signs, device=device)


def _mirror_joints(values, permutation, signs):
    return values[..., permutation] * signs


def mirror_observations(obs, env, obs_type):
    permutation, signs = _joint_mirror(env, obs.device)
    n = len(permutation)
    out = obs.clone()
    polar_xyz = torch.tensor([1.0, -1.0, 1.0], device=obs.device)
    axial_xyz = torch.tensor([-1.0, 1.0, -1.0], device=obs.device)
    polar_xy = torch.tensor([1.0, -1.0], device=obs.device)

    if obs_type == "policy":
        # gravity, angular velocity, camera ball XY, target direction XY
        out[..., 0:3] *= polar_xyz
        out[..., 3:6] *= axial_xyz
        out[..., 6:8] *= polar_xy
        out[..., 8:10] *= polar_xy
        offset = 13  # visibility, age, confidence remain unchanged
        for _ in range(3):  # joint position, velocity, previous action
            out[..., offset : offset + n] = _mirror_joints(
                obs[..., offset : offset + n], permutation, signs
            )
            offset += n
    elif obs_type == "critic":
        # gravity, base linear velocity, base angular velocity, base height
        out[..., 0:3] *= polar_xyz
        out[..., 3:6] *= polar_xyz
        out[..., 6:9] *= axial_xyz
        # exact ball XYZ, ball-relative velocity XY
        out[..., 10:13] *= polar_xyz
        out[..., 13:15] *= polar_xy
        # ball displacement is scalar; target direction is XY; alignment is scalar
        out[..., 16:18] *= polar_xy
        feet = obs[..., 19:25].reshape(*obs.shape[:-1], 2, 3)
        out[..., 19:25] = feet.flip(-2).mul(polar_xyz).reshape(*obs.shape[:-1], 6)
        offset = 25
        for _ in range(3):
            out[..., offset : offset + n] = _mirror_joints(
                obs[..., offset : offset + n], permutation, signs
            )
            offset += n
    else:
        raise ValueError(f"Unsupported observation type: {obs_type!r}")

    if offset != obs.shape[-1]:
        raise ValueError(
            f"Adjust {obs_type} symmetry layout expected {offset} features, got {obs.shape[-1]}"
        )
    return out


def mirror_actions(actions, env):
    permutation, signs = _joint_mirror(env, actions.device)
    return _mirror_joints(actions, permutation, signs)


def data_augmentation_func(obs, actions, env, obs_type):
    mirrored_obs = None if obs is None else mirror_observations(obs, env, obs_type)
    mirrored_actions = None if actions is None else mirror_actions(actions, env)
    augmented_obs = None if obs is None else torch.cat((obs, mirrored_obs), dim=0)
    augmented_actions = None if actions is None else torch.cat((actions, mirrored_actions), dim=0)
    return augmented_obs, augmented_actions

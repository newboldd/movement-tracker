"""Differentiable joint angle constraint loss using palm-normal decomposition.

Computes flex/abd angles using the *exact same* math as
``skeleton_data._compute_angles`` (the function that produces the plotted traces),
then penalises angles outside the per-joint [min, max] ranges from the priors
JSON.  Both fitting algorithms (v1, v2) import the single entry-point
``compute_angle_constraint_loss``.
"""
from __future__ import annotations

import torch

# Joints on the thumb kinematic chain (use thumb_ref instead of palm_n)
_THUMB_JOINTS = {1, 2, 3}


def _tnorm(v: torch.Tensor) -> torch.Tensor:
    """Row-wise normalise (N,3), zero-length rows become zero."""
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return v / n


def _talign(n: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Flip rows of *n* whose dot with *ref* is negative."""
    d = (n * ref).sum(-1, keepdim=True)
    return torch.where(d >= 0, n, -n)


_JOINT_GROUP_MAP = {
    "Thumb CMC": "thumb_cmc_mcp", "Thumb MCP": "thumb_cmc_mcp",
    "Thumb IP": "thumb_ip",
    "Index MCP": "finger_mcp", "Middle MCP": "finger_mcp",
    "Ring MCP": "finger_mcp", "Pinky MCP": "finger_mcp",
    "Index PIP": "finger_pip_dip", "Index DIP": "finger_pip_dip",
    "Middle PIP": "finger_pip_dip", "Middle DIP": "finger_pip_dip",
    "Ring PIP": "finger_pip_dip", "Ring DIP": "finger_pip_dip",
    "Pinky PIP": "finger_pip_dip", "Pinky DIP": "finger_pip_dip",
}


def compute_angle_constraint_loss(
    joints_3d: torch.Tensor,       # (N, 21, 3)
    priors: list[dict],            # entries from priors JSON "joints" array
    constraint_groups: dict | None = None,  # group_name → {"flex": bool, "abd": bool}
) -> torch.Tensor:
    """Return a scalar loss penalising angles outside their prior ranges.

    Each prior dict must have keys:
        parent, joint, child,
        flex_min, flex_max, flex_std,
        abd_min, abd_max, abd_std
    """
    device = joints_3d.device
    loss = torch.tensor(0.0, device=device)

    if not priors:
        return loss

    # ── Palm normal (robust average of 4 cross-products) ───────────────
    w = joints_3d[:, 0]
    e_idx  = joints_3d[:, 5]  - w
    e_mid  = joints_3d[:, 9]  - w
    e_ring = joints_3d[:, 13] - w
    e_pnk  = joints_3d[:, 17] - w

    n_ip = _tnorm(torch.cross(e_idx,  e_pnk, dim=1))
    n_im = _talign(_tnorm(torch.cross(e_idx,  e_mid, dim=1)),  n_ip)
    n_mr = _talign(_tnorm(torch.cross(e_mid,  e_ring, dim=1)), n_ip)
    n_rp = _talign(_tnorm(torch.cross(e_ring, e_pnk, dim=1)),  n_ip)
    palm_n = _tnorm(n_ip + n_im + n_mr + n_rp)

    # ── Thumb reference: toward pinky MCP ─────────────────────────────
    pinky_dir = _tnorm(e_pnk)
    thumb_ref = pinky_dir

    # ── MCP local dorsal axes: cross product of adjacent inter-MCP segments ──
    # MCP local dorsal axes — sign determined once from median alignment with palm normal
    _MCP_NEIGHBORS = {
        5:  (1, 9),    # I_MCP: between T_CMC and M_MCP
        9:  (5, 13),   # M_MCP: between I_MCP and R_MCP
        13: (9, 17),   # R_MCP: between M_MCP and P_MCP
    }
    mcp_local_ref: dict[int, torch.Tensor] = {}
    for mcp_j, (na, nb) in _MCP_NEIGHBORS.items():
        va = _tnorm(joints_3d[:, na] - joints_3d[:, mcp_j])
        vb = _tnorm(joints_3d[:, nb] - joints_3d[:, mcp_j])
        local_n = _tnorm(torch.cross(va, vb, dim=1))
        # Sign from median dot with palm normal (stable, no per-frame flipping).
        # Align opposite to palm_n: palm_n points dorsal, we want palmar.
        median_dot = (local_n * palm_n).sum(-1).median().item()
        if median_dot > 0:
            local_n = -local_n
        mcp_local_ref[mcp_j] = local_n
    # Pinky MCP
    va_pnk = _tnorm(joints_3d[:, 13] - joints_3d[:, 17])
    vb_pnk = _tnorm(joints_3d[:, 0]  - joints_3d[:, 17])
    local_n_pnk = _tnorm(torch.cross(va_pnk, vb_pnk, dim=1))
    median_dot_pnk = (local_n_pnk * palm_n).sum(-1).median().item()
    if median_dot_pnk > 0:
        local_n_pnk = -local_n_pnk
    mcp_local_ref[17] = local_n_pnk

    # ── Per-joint cache for reference propagation (PIP/DIP) ────────────
    #    cache[j] = (dorsal_axis, b_in)   — both (N, 3)
    da_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    PI = 3.141592653589793

    for pr in priors:
        p, j, c = pr["parent"], pr["joint"], pr["child"]

        b_in  = _tnorm(joints_3d[:, j] - joints_3d[:, p])
        b_out = _tnorm(joints_3d[:, c] - joints_3d[:, j])

        # Choose reference: MCP local dorsal for finger MCPs, thumb_ref for thumb, palm_n fallback
        if p == 0 and j in mcp_local_ref:
            ref = mcp_local_ref[j]
        elif p == 0:
            ref = thumb_ref if j in _THUMB_JOINTS else palm_n
        elif p in da_cache:
            da_par, bi_par = da_cache[p]
            sin_t = (-(b_in * da_par).sum(-1, keepdim=True)).clamp(-1, 1)
            cos_t = (1 - sin_t ** 2).clamp(min=0).sqrt()
            ref = sin_t * bi_par + cos_t * da_par
        else:
            ref = root_ref

        # Dorsal axis = ref projected ⊥ b_in
        dorsal_axis = _tnorm(ref - (ref * b_in).sum(-1, keepdim=True) * b_in)
        flex_axis   = _tnorm(torch.cross(dorsal_axis, b_in, dim=1))

        # Cache for child joints
        da_cache[j] = (dorsal_axis, b_in)

        # ── Flex angle (same convention as _compute_angles) ────────────
        flex_deg = torch.atan2(
            -(b_out * dorsal_axis).sum(-1),
            (b_out * b_in).sum(-1),
        ) * (180.0 / PI)

        # ── Abd angle ──────────────────────────────────────────────────
        flex_rad = flex_deg * (PI / 180.0)
        bfe = flex_rad.unsqueeze(-1).cos() * b_in - flex_rad.unsqueeze(-1).sin() * dorsal_axis
        bfe = _tnorm(bfe)

        abd_deg = torch.atan2(
            (b_out * flex_axis).sum(-1),
            (b_out * bfe).sum(-1),
        ) * (180.0 / PI)

        # ── Penalise out-of-range (skip if group is disabled) ──────
        joint_name = pr.get("name", "")
        group_name = _JOINT_GROUP_MAP.get(joint_name)
        grp = (constraint_groups or {}).get(group_name, {"flex": True, "abd": True}) if group_name else {"flex": True, "abd": True}

        fl, fh = pr["flex_min"], pr["flex_max"]
        al, ah = pr["abd_min"],  pr["abd_max"]

        if grp.get("flex", True):
            f_below = torch.clamp(fl - flex_deg, min=0)
            f_above = torch.clamp(flex_deg - fh, min=0)
            loss = loss + (f_below ** 2 + f_above ** 2).mean()

        if grp.get("abd", True):
            a_below = torch.clamp(al - abd_deg, min=0)
            a_above = torch.clamp(abd_deg - ah, min=0)
            loss = loss + (a_below ** 2 + a_above ** 2).mean()

    return loss

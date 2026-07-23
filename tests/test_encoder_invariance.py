"""Layer-level geometric-invariance tests for the MolConv encoder versions.

Feeds a synthetic point cloud through the first encoder layer (remove_xyz=True)
with random weights and checks how the per-atom output responds to rigid
transforms. Invariance is an architectural property, so random weights suffice.

Claimed / intended properties
-----------------------------
  v1 (MolConv1) : absolute-position Gram -> rotation + reflection + permutation
                  invariant, but NOT translation invariant (legacy).
  v2 (MolConv2) : relative-displacement Gram + centering + padding-mask ->
                  full E(3) (rotation + reflection + translation) + permutation.
  v2 + chirality (SE(3)) : rotation + translation invariant, but reflection-
                  SENSITIVE by design (chiral tasks, e.g. 3DMolCSP).
"""
import numpy as np
import torch

from molnetpack.molconv import MolConv1, MolConv2

B, IN, P, K, OUT, NR = 2, 21, 300, 5, 64, 20
DTYPE = torch.float64
INVAR = 1e-8      # invariant threshold (float64, architectural)
SENS = 1e-2       # "clearly sensitive" threshold


def _synthetic():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(B, IN, P, dtype=DTYPE)
    x[:, :3, :NR] = torch.randn(B, 3, NR, generator=g, dtype=DTYPE) * 3.0
    x[:, 3:, :NR] = torch.randn(B, IN - 3, NR, generator=g, dtype=DTYPE)
    mask = torch.zeros(B, P, dtype=torch.bool); mask[:, :NR] = True
    idx_base = torch.arange(0, B).view(-1, 1, 1) * P
    return x, mask, idx_base


def _rot_mat():
    g = torch.Generator().manual_seed(1)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DTYPE))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rot(x, Q):
    z = x.clone(); z[:, :3, :] = torch.einsum('ij,bjp->bip', Q, z[:, :3, :]); return z
def _refl(x):
    z = x.clone(); z[:, 0, :] *= -1; return z
def _trans(x):
    z = x.clone(); z[:, :3, :NR] += torch.tensor([1.7, -2.3, 0.9], dtype=DTYPE).view(1, 3, 1); return z
def _perm(x):
    z = x.clone(); p = torch.randperm(NR); z[:, :, :NR] = z[:, :, p]; return z


def _rel(base, out):
    """Max abs diff over real atoms, relative to output scale."""
    d = (base[:, :, :NR] - out[:, :, :NR]).abs().max().item()
    return d / (base[:, :, :NR].abs().mean().item() + 1e-12)


def _run(layer, x, mask, idx_base, needs_mask):
    return layer(x, idx_base, mask) if needs_mask else layer(x, idx_base)


def _deltas(layer, needs_mask):
    layer = layer.eval().double()
    x, mask, idx_base = _synthetic()
    Q = _rot_mat()
    with torch.no_grad():
        base = _run(layer, x, mask, idx_base, needs_mask)
        out = {
            "rotation":    _run(layer, _rot(x, Q), mask, idx_base, needs_mask),
            "reflection":  _run(layer, _refl(x), mask, idx_base, needs_mask),
            "translation": _run(layer, _trans(x), mask, idx_base, needs_mask),
            "permutation": _run(layer, _perm(x), mask, idx_base, needs_mask),
        }
    # permutation compares the pooled (set) output; others compare matched atoms
    rel = {k: _rel(base, v) for k, v in out.items() if k != "permutation"}
    dp = (base[:, :, :NR].mean(2) - out["permutation"][:, :, :NR].mean(2)).abs().max().item()
    rel["permutation"] = dp / (base[:, :, :NR].abs().mean().item() + 1e-12)
    return rel


def _mc2(**flags):
    torch.manual_seed(42)
    return MolConv2(IN, OUT, P, K, remove_xyz=True, **flags)


def test_v2_is_e3_and_permutation_invariant():
    """v2 (production encoder) is invariant to rotation, reflection, translation, permutation."""
    d = _deltas(_mc2(), needs_mask=True)
    for g in ("rotation", "reflection", "translation", "permutation"):
        assert d[g] < INVAR, f"v2 not invariant to {g}: relΔ={d[g]:.2e}"


def test_v2_chirality_is_se3_reflection_sensitive():
    """chirality=True must keep rotation+translation invariance but BREAK reflection (SE(3))."""
    d = _deltas(_mc2(chirality=True), needs_mask=True)
    assert d["rotation"] < INVAR,     f"chirality broke rotation: {d['rotation']:.2e}"
    assert d["translation"] < INVAR,  f"chirality broke translation: {d['translation']:.2e}"
    assert d["permutation"] < INVAR,  f"chirality broke permutation: {d['permutation']:.2e}"
    assert d["reflection"] > SENS,    f"chirality (SE3) should be reflection-sensitive: {d['reflection']:.2e}"


def test_v1_is_o3_and_permutation_but_not_translation():
    """Legacy v1 (absolute Gram): rotation/reflection/permutation invariant, NOT translation."""
    d = _deltas(MolConv1(IN, OUT, K, remove_xyz=True), needs_mask=False)
    assert d["rotation"] < INVAR,    f"v1 rotation: {d['rotation']:.2e}"
    assert d["reflection"] < INVAR,  f"v1 reflection: {d['reflection']:.2e}"
    assert d["permutation"] < INVAR, f"v1 permutation: {d['permutation']:.2e}"
    assert d["translation"] > INVAR, f"v1 should NOT be exactly translation-invariant: {d['translation']:.2e}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")

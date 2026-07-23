# Intro to MolConv

`MolConv` is the graph-convolution layer at the heart of 3DMolMS. It turns a
molecule's **3D conformation** — a cloud of atoms, each with `xyz` coordinates and
chemical features — into a fixed-length embedding that the task heads (MS/MS, RT,
CCS) read to make their predictions.

Concretely, the encoder takes up to `max_atom_num` atoms, each a 21-dimensional
vector (the first 3 dims are `xyz`; the remaining 18 are one-hot atom features),
and returns one embedding per molecule.

## Why invariance matters

The same molecule can be written down in infinitely many ways: shift it a few
ångström, rotate it, mirror it, or list its atoms in a different order — it is
still the *same molecule* with the *same chemistry*. A good encoder must return
the **same embedding** under all of these, otherwise the prediction would depend
on arbitrary details of how the conformer happened to be stored.

| Transformation | Example | Should the embedding change? |
|---|---|---|
| **Permutation** | reorder the atom list | no |
| **Rotation** | spin the molecule | no |
| **Translation** | move it in space | no |
| **Reflection** | mirror it (its enantiomer) | usually no — but *yes* for chiral tasks |

These combine into the named symmetry groups used throughout this page: **O(3)** =
rotation + reflection; **SE(3)** = rotation + translation; **E(3)** = rotation +
reflection + translation. For the achiral 3DMolMS tasks the target is
**E(3) + permutation** invariance — ignore *where* and *how* the molecule sits in
space, but still see its shape.

## Two versions

- **v1** (`MolConv1`) — the original layer.
- **v2** (`MolConv2`) — the corrected, fully E(3)-invariant layer.

Select with `encoder_version: 1|2` in the model config. The bare name **`MolConv`**
is an alias for `MolConv2`, so code that imports `MolConv` uses the E(3) encoder (v2).

## Summary

| | **v1** (`MolConv1`) | **v2** (`MolConv2`) |
|---|---|---|
| First-layer Gram matrix | absolute `⟨xⱼ, xₗ⟩`, **uncentered** | relative `⟨xⱼ−xᵢ, xₗ−xᵢ⟩` on the **centered** frame |
| Padding atoms in kNN | included (sit at the origin) | **masked out** of real atoms' neighborhoods |
| Coordinate centering | none | on the real-atom centroid |
| Normalization | `BatchNorm2d` | `LayerNorm` |
| Rotation invariant | ✅ | ✅ |
| Reflection invariant | ✅ | ✅ |
| **Translation invariant** | ❌ | ✅ |
| Permutation invariant | ✅ | ✅ |
| **Symmetry group** | **O(3) + permutation** | **E(3) + permutation** |
| Params (RT model) | 11.7 M | 21.5 M |

The invariance rows are measured, not claimed — see [Invariance
verification](#invariance-verification).

## v1 — `MolConv1`

**In short: v1's geometry depends on where the molecule sits in space, so moving
the molecule silently changes its embedding.**

The first layer builds its Gram matrix from **absolute atom positions**
`⟨xⱼ, xₗ⟩` (via `graph_feat @ graph_featᵀ`) using the raw, uncentered
coordinates, and its k-nearest-neighbour graph includes the zero-padding atoms
(which all sit at the origin). Both choices tie the output to the molecule's
absolute pose:

- A rigid **translation** of the conformer changes the absolute positions and the
  real→padding distances, so the embedding changes. Measured shift: ~**11–18 %**
  of the output magnitude under a modest translation.
- Rotation, reflection and permutation *are* preserved (inner products and
  distances are orthogonally invariant; padding-at-origin is rotation-symmetric).

So v1 is **O(3) + permutation invariant but not translation-invariant** — it does
**not** satisfy SE(3)/E(3). Because conformers are stored pre-centered, this
fragility is mostly latent in practice, but it is a real correctness gap: a shift
of the input silently changes the prediction. v1 is retained only for reproducing
existing checkpoints.

## v2 — `MolConv2`

**In short: v2 describes geometry through relative positions and angles rather
than absolute coordinates, so the molecule's pose no longer matters.**

Three changes make it genuinely pose-independent:

1. **Relative-displacement Gram** `⟨xⱼ−xᵢ, xₗ−xᵢ⟩` — this inner product encodes the
   local angle `∠jik` between neighbours, which is unchanged by rotation and
   reflection.
2. **Center on the real-atom centroid** before computing the first-layer geometry.
   The centroid moves with any translation of the input, so the centered
   coordinates (and all distances) are identical under a shift → translation
   invariance becomes exact.
3. **Mask zero-padding atoms out of real atoms' kNN.** After centering, padding
   sits at the centroid and would otherwise pollute real atoms' neighborhoods and
   leak a pose-dependent term through the atom-dimension normalization; masking
   keeps those padding rows inert.

The result is **full E(3) + permutation invariance** (rotation + reflection +
translation), verified to floating-point-exact precision in `float64`. v2 is the
production encoder for all released checkpoints.

## Performance

Across the downstream tasks (MS/MS, RT, CCS), **v2 matches or beats v1 on accuracy
and is the only E(3)-invariant option** — the switch to v2 is a correctness
improvement at no accuracy cost.

## Chirality: E(3) (default) vs SE(3)

Reflection is the one symmetry you sometimes *don't* want. By default both v1 and
v2 are **reflection-invariant**, so a molecule and its mirror image (its
enantiomer) get the **identical** embedding — the encoder cannot tell enantiomers
apart. That is exactly right for the **achiral 3DMolMS tasks** (MS/MS, RT, CCS),
where enantiomers share the same target.

For a **chirality-dependent** task — e.g. **3DMolCSP** (chiral stationary-phase
separation), where enantiomers elute differently — set **`chirality: true`** on
v2. It appends a **signed-volume pseudoscalar** channel `dⱼ·(d₁×d₂)` in the first
layer, which is invariant under proper rotation and translation but **flips sign
under reflection** — so the encoder becomes **SE(3)** (reflection-*sensitive*).

| | `chirality: false` (default) | `chirality: true` |
|---|---|---|
| Symmetry group | **E(3)** | **SE(3)** |
| Rotation | invariant | invariant |
| Translation | invariant | invariant |
| **Reflection** | **invariant** (Δ = 0, exact) | **sensitive** (Δ ≈ 0.4) |
| Enantiomers (mirror images) | identical prediction | **can differ** |
| Correct for | achiral tasks: MS/MS, RT, CCS | chiral tasks: e.g. 3DMolCSP |

An E(3) encoder is **provably unable** to model enantiomer differences (it must
output one value for both), so SE(3) is *required* for chiral separation — not
optional. The reflection responses above are measured on random-weight layers
(see `test_v2_chirality_is_se3_reflection_sensitive`).

## Invariance verification

Invariance here is **architectural** — it holds for any weights — so it can be
checked with random-weight layers, with no trained checkpoint needed:

- `tests/test_encoder_invariance.py` — layer-level rotation/reflection/
  translation/permutation checks for v1, v2, and the `chirality` (SE(3)) mode.
- `tests/test_se3_invariance.py` — full-encoder SE(3) invariance in `float64` and
  `float32`, including the at-origin regression test for the padding mask.

```bash
pytest tests/test_encoder_invariance.py tests/test_se3_invariance.py -q
```

## Configuration

```yaml
model:
  encoder_version: 2   # 1 = MolConv1 (O(3)); 2 = MolConv2 (E(3); also aliased as MolConv)
  chirality: false     # true -> SE(3), reflection-sensitive (chiral tasks, e.g. 3DMolCSP)
```

Both versions share the same `state_dict` key layout, so switching
`encoder_version` changes the forward computation, **not** the parameter shapes —
but a checkpoint must be evaluated with the same settings it was trained under.

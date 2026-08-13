# CBraMod Integration Contract

This document defines the model/code identity and the validated boundary
between frozen EEG manifests and CBraMod. The main classifier family and
plasticity depth are locked.

## Pinned identity

- Official code: <https://github.com/wjq-learning/CBraMod>
- Code commit: `b9e961003214326972c567eff390e75b0287e32a`
- Official checkpoint: <https://huggingface.co/weighting666/CBraMod>
- Checkpoint revision: `500543c7e30bda1b22bfd51a49301b238dee21fd`
- Checkpoint file SHA-256:
  `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`
- Checkpoint bytes: `19,775,842`

Checkpoint loading uses `torch.load(..., weights_only=True)` followed by strict
state-dict loading. A mismatch in file identity or architecture fails before a
run starts. The ignored local copy lives at
`checkpoints/cbramod/pretrained_weights.pth` and can be materialized by
`scripts/download_cbramod_checkpoint.py`.

The adapted architecture retains the upstream state-dict names and equations.
On a fixed random `1x2x3x200` tensor, it produced bit-identical output to the
pinned upstream checkout (`max_abs_diff = 0.0`). The upstream code is MIT
licensed; the preserved notice is at `LICENSES/CBraMod-MIT.txt`. The Hugging
Face checkpoint repository declares Apache-2.0.

## Shape and plasticity boundary

The backbone accepts variable channel and patch dimensions while retaining
200-point patches:

- BCI IV-2a and PhysioNet-MI: `B x 22 x 4 x 200`.
- Sleep-EDF: `B x 2 x 30 x 200`.
- Conditional TUEV: `B x 16 x 5 x 200`.

No pad-and-mask workaround is needed for these shapes. Criss-cross attention
derives channel and patch dimensions at runtime. `set_trainable_depth(N)`
freezes patch embedding and all but the last `N` encoder blocks, matching the
predeclared 1/2/4/8-block sweep.

The main task head is the upstream all-patch family: flatten all channel-patch
representations, then `Linear(C*P*200, 800) -> ELU -> Dropout -> Linear(800,
200) -> ELU -> Dropout -> Linear(200, classes)`. Its task-specific parameter
count is reported separately from shared-backbone memory.

The Week-1 linear probe also sees every channel-patch representation but has
only one linear output layer. The earlier mean-pool linear run is retained as a
negative control because averaging first discarded the structure used by the
main head. It is not used for the fine-tune-versus-probe gate.

When only the final `N` blocks are plastic, the patch embedding and preceding
frozen encoder blocks remain in eval mode. This matters because they contain
dropout: enabling training mode in frozen blocks made the representation
stochastic and strongly biased the first depth sweep against shallow depths.
That sweep is explicitly invalidated in config; depth v3 is the first eligible
selection run. Depth v3 selected the final four encoder blocks: its normalized
mean validation BA was 0.4335 versus the best 0.4473 at depth 8, clearing the
predeclared 95% threshold of 0.4249. See decision 0001 for per-task evidence.

## Executable proof

`scripts/smoke_test_cbramod_gpu.py` loads real samples only through manifest v3,
strict-loads the pretrained checkpoint for each task, unfreezes the requested
last encoder blocks, and requires finite non-zero backbone gradients after
cross-entropy backward.

Observed on one NVIDIA L40S with PyTorch `2.13.0+cu130`, batch size 2 and depth
1:

| Dataset | Input | Features | Logits | Loss | Backbone grad norm |
|---|---:|---:|---:|---:|---:|
| BCI IV-2a A01E | `2x22x4x200` | `2x22x4x200` | `2x4` | 1.39894 | 0.80452 |
| PhysioNet S001R04 | `2x22x4x200` | `2x22x4x200` | `2x4` | 1.37387 | 0.81724 |
| Sleep SC00 night 1 | `2x2x30x200` | `2x2x30x200` | `2x5` | 1.63423 | 1.52974 |

All three paths propagated gradients through exactly 402,600 parameters in the
last criss-cross block. These random-head losses are connectivity proof, not
performance results.

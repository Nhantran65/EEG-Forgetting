# Frozen manifests

Paper-facing dataset and split manifests live here after they have been created
from an audited local dataset snapshot. Manifest files are create-only: generate
a new version instead of overwriting one referenced by a run.

`v1/` freezes:

- BCI IV-2a at 5/2/2 subjects.
- PhysioNet-MI at 70/18/17 clean subjects plus four explicit exclusions, and
  the one-off 70/19/20 reproduction assignment.
- Sleep-EDF at 48/15/15 subjects, stratified by age band and sex with seed
  `20260813`; all nights from one subject stay together.

`manifest-set.json` records the row count and SHA-256 of each JSONL file plus
the dataset-config hashes. Every JSONL row records source-file SHA-256 values.
Run `scripts/audit_manifests.py --verify-sources` to verify the complete chain.

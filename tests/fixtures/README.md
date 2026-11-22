# Test fixtures

These fixtures contain no videos, model weights, images of people, or plate text.

- `audit_tasks_synthetic.json` supplies invented task intervals for testing the
  existing evaluation-clip-to-parent frame mapping.
- `plate_annotation_synthetic_recipe.json` expands in pytest temporary storage
  into 600 synthetic geometry records. It exercises dense-frame completeness,
  annotation provenance, coordinate comparisons, export hashes, and the CLI.
  The v1 validator requires specific frozen package/source identities, so the
  tests retain those metadata fields from the tracked Ground Truth. The fake
  boxes and simulated review states are test inputs, not benchmark or human
  review evidence. The frozen Ground Truth files are never modified.

The default suite validates the tracked published Ground Truth structure and
content hash without loading private media. The separate `local_assets` test
validates the actual annotation export and source-media hashes only when
`REDACTOR_RUN_LOCAL_ASSET_TESTS=1` is explicitly set.

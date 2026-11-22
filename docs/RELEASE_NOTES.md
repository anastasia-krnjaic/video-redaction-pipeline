# Release Notes

## Release Summary

This release freezes the local video redaction pipeline through product Final publication, delivery packaging, evaluation disposition, production-readiness optimization, cleanup, and documentation.

The current submission release label is:

```text
submission-release-v1.0.3
```

## Major Capabilities

- Unified face and license-plate automatic preview workflow.
- Shared privacy rendering contract for accepted regions.
- Source-bound Final Review Workspace for Day and Night cases.
- Advisory Guidance model with fail-closed Needs correction.
- Entity policy workflow for people and plates.
- Manual correction Preview, Save, and individual correction removal with real rendered privacy output.
- Explicit session routing, case-safe decision persistence, and stale-publication protection.
- Frame-accurate seeking, cached entity previews, entity conflict handling, and human-confirmed candidate merges.
- Immutable Latest reviewed revisions.
- Revision-bound full-video approval.
- Immutable case-scoped Final media.
- Portable delivery packages.
- Frozen detector, automatic, and reviewed-final evaluation disposition.
- Single-writer, atomic-state, resumable, and telemetry-backed local operation.

## Current Approved Finals

| Case | Latest reviewed revision | Finalization ID | Final media SHA-256 |
| --- | --- | --- | --- |
| Day | `latest_reviewed_4808aa899c5da1a8` | `finalization_a9cc6da4ac152e74` | `4f0c57731b4d99ed2b4397472c8d991d3debb9962de1b4419cb941cff296d837` |
| Night | `latest_reviewed_41fcfdd5b7e2349d` | `finalization_2e7257daf1187805` | `cc8982b317c439d7965dd151c562f0c01004b3252cfd5f63266fa95b0a21e88a` |

Both current Final videos were watched through in full by the reviewer, approved at the exact Latest reviewed revision, and then published as immutable Finals.

## Current Full Final Packages

| Package | SHA-256 |
| --- | --- |
| Day | `675a9a9362350824e10bbd941cebf20f041b5237c74f68300156482d1da5791a` |
| Night | `396a51bdbf4988be7e128be35c61de3190896abb0376b5ccc7b4fea593ef023d` |

## Evaluation Disposition

- Plate detector aggregate recall: 557 / 600 = 0.928333.
- Automatic eligible aggregate coverage: 240 / 600 = 0.4.
- Day reviewed-final plate metric: unavailable due source incompatibility.
- Historical Night reviewed-final strict geometric coverage: 284 / 360 = 0.788889, bound to `latest_reviewed_133bd8a1e52cfa36` and media SHA-256 `5a010c26f5f3367f14a1fc25327ec93e7f244bf8c22b760946140dd928968e60`.
- The 76-frame visual audit belongs to that same historical revision. Neither result is assigned to current Final revision `latest_reviewed_41fcfdd5b7e2349d`; see [Evaluation](EVALUATION.md).
- Face reviewed-final numerical scoring is not supported by the frozen face evaluation contract.
- The Night submission sample covers source frames 484-686 inclusive, is 203 frames/about 6.77 seconds, and has separate machine validation plus human visual acceptance with no known visible sensitive-data leak observed.

See [EVALUATION.md](EVALUATION.md).

## Production Hardening

The release includes single-writer protection, atomic event and pointer state, projection and immutable metadata caching, no-op materialization, heavy-job admission control, resumable stages, telemetry, streaming validation, and artifact cleanup.

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

## Known Limitations

- The repository source checkout does not include private source media, model weights, review sessions, Final media, or delivery packages.
- Day reviewed-final plate scoring is not comparable to the frozen `day_plate` benchmark source.
- Face reviewed-final numerical scoring is not available under the frozen contract.
- The local server is not a remote multi-user deployment.

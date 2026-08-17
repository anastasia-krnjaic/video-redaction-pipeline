# Release Notes

## Release Summary

This release freezes the local video redaction pipeline through product Final publication, delivery packaging, evaluation disposition, production-readiness optimization, cleanup, and documentation.

The release tag recommendation is:

```text
release-v1.0
```

## Major Capabilities

- Unified face and license-plate automatic preview workflow.
- Shared privacy rendering contract for accepted regions.
- Source-bound Final Review Workspace for Day and Night cases.
- Advisory Guidance model with fail-closed Needs correction.
- Entity policy workflow for people and plates.
- Manual correction Preview and Save with real rendered privacy output.
- Immutable Latest reviewed revisions.
- Revision-bound full-video approval.
- Immutable case-scoped Final media.
- Portable delivery packages.
- Frozen detector, automatic, and reviewed-final evaluation disposition.
- Single-writer, atomic-state, resumable, and telemetry-backed local operation.

## Frozen Finals

| Case | Final media SHA-256 |
| --- | --- |
| Day | `f2c16de58e145a7c5073e6fdbf9af9b6c39a9033ce1034a59099901d6333d8a0` |
| Night | `5a010c26f5f3367f14a1fc25327ec93e7f244bf8c22b760946140dd928968e60` |

## Frozen Delivery Packages

| Package | SHA-256 |
| --- | --- |
| Day | `dd20170dd1674899051b42ef36806af0c6160acf265afa73fa4f84af0f006eb0` |
| Night | `a17a4f7f3bc5bad4eb99ddc61f89c05c44a0c608ca8498ed92ef88d484c98286` |
| Combined | `dfc71335b72d7e9201fed1407840e2b03aa5f87e9a95cf8cac94e87469437f69` |

## Evaluation Disposition

- Plate detector aggregate recall: 557 / 600 = 0.928333.
- Automatic eligible aggregate coverage: 240 / 600 = 0.4.
- Day reviewed-final plate metric: unavailable due source incompatibility.
- Night reviewed-final strict geometric coverage: 284 / 360 = 0.788889.
- Night visual audit of the 76 strict geometric misses found no true visible privacy leaks.
- Face reviewed-final numerical scoring is not supported by the frozen face evaluation contract.

See [EVALUATION.md](EVALUATION.md).

## Production Hardening

The release includes single-writer protection, atomic event and pointer state, projection and immutable metadata caching, no-op materialization, heavy-job admission control, resumable stages, telemetry, streaming validation, and artifact cleanup.

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

## Known Limitations

- The repository source checkout does not include private source media, model weights, review sessions, Final media, or delivery packages.
- Day reviewed-final plate scoring is not comparable to the frozen `day_plate` benchmark source.
- Face reviewed-final numerical scoring is not available under the frozen contract.
- The local server is not a remote multi-user deployment.

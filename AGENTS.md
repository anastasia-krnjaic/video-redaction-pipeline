# Repository Guidance

## System Overview Maintenance

`docs/SYSTEM_OVERVIEW.md` is the canonical human-readable description of the current pipeline. Any task that materially changes the pipeline must update `docs/SYSTEM_OVERVIEW.md` as part of the same change. Before changing the canonical overview, preserve the previous canonical document as the next sequential immutable file under `docs/system_overview_versions/`, then update `docs/system_overview_versions/CHANGELOG.md`.

Every archived version must preserve the implementation sources and configuration sources on which it was based. Archived versions must never be edited retroactively. Do not create a new archived version for spelling, grammar, or formatting corrections that do not change the documented system behavior.

A material system change includes any change to detector model or model version, model selection status, confidence thresholds, shared detection logic, track association, track confirmation criteria, interpolation or prediction rules, analyst decisions or manual override workflow, identity grouping, rendering eligibility, redaction rendering, audit records or manifests, output artifacts, evaluation methodology, benchmark clips, or production workflow.

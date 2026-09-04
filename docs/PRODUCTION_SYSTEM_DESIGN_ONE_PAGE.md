# Production System Design One-Page

## Architecture / Workflow

The system is a source-bound video redaction workflow. Each case begins by ingesting the source video, extracting FFprobe metadata, and recording a source hash. The automated path then runs the supported face and license-plate inference stages, associates detections into temporal tracks, applies deterministic/bounded recovery for defensible short gaps, and renders an immutable **Automatic** video. Automatic is evidence for review, not the final privacy claim.

Human review happens in the local Final Review Workspace. Reviewers inspect Guidance, apply privacy-conservative Entity policy, add or retract Manual corrections when needed, preview corrections on a local short frame or interval, save one or more pending changes, and explicitly select **Build reviewed video**. The build compiles the current reviewed state into a validated **Latest reviewed** revision. Final publication requires a full-video approval bound to that exact Latest reviewed revision, then creates an immutable **Final** video and delivery package.

## Inference / GPU Efficiency

The current release is a local implementation, not a distributed GPU service. It uses the selected OpenVINO ADAS face detector for `face` and the OIM YOLOv9-S-608 model for `license_plate`, with category-specific tracking/recovery semantics preserved. Production deployment would run detector workloads on GPU-capable workers using batched or streamed frame inference, separate detector work from decode/render work, and reuse immutable stage outputs by source hash, model/config identity, and processing span. Review edits should not trigger detector reruns. Zero-change reviewed builds can reuse validated Automatic media instead of running a heavy full-video encode.

## Human In The Loop

Guidance highlights intervals that deserve reviewer attention but does not fabricate Correct decisions and does not alone block finalization. Needs correction is fail-closed. The player supports frame-accurate seeking in source-frame coordinates. Entity actions apply to the whole source-backed person or plate, use on-demand/background-cached previews, surface purity/conflict risks, and require human confirmation for candidate merges. Manual Preview verifies draft geometry locally; Save or Remove persists authoritative review state; **Build reviewed video** materializes all pending media-affecting changes in one job. Full-video approval remains the final human gate before Final.

## Storage / Provenance

Portable evidence is built from source hashes, source-frame coordinates, model/config identities, append-only review events, reviewed-region records, revision pointers, media hashes, and validation artifacts. Reviewed-region records preserve category, frame/timestamp basis, geometry, entity/track binding, provenance, detector evidence where available, manual correction identity or retraction where applicable, reviewed-region-set identity, and output revision binding. Delivery packages include media, summary manifests, validation state, and packaged reviewed-region records with hashes so the per-redaction record can be verified without local absolute paths.

## Queueing / Concurrency

The local release uses explicit session routing, a single-writer workspace lock, case-scoped decision persistence, and background admission for heavy reviewed-video materialization. Media-affecting edits mark Latest reviewed as pending. Explicit build jobs are guarded by the active review-state projection; stale or superseded jobs cannot publish current media. A production service would move this to durable queues, worker leases, idempotent job keys, retry/dead-letter handling, and per-case concurrency controls.

## Bottlenecks

The expensive stages are detector inference, video decode/encode, large media I/O, and full-span validation scans. Review-state projection and audit manifest generation are comparatively small. Production monitoring should track GPU utilization, queue latency, encode time, validation time, storage throughput, artifact cache hit rate, and human review time per case.

## Scale / Cost Tradeoffs

Higher GPU batch sizes improve utilization but can increase per-case latency. Object storage and immutable artifact reuse reduce rerun cost but require lifecycle, retention, and deduplication policy. A durable database/event store improves concurrent review and recovery but adds operational complexity. Horizontal case workers scale throughput, while full-video review remains a human-cost bottleneck. Retaining full Original/Automatic/Latest/Final media improves auditability but increases storage cost.

## Limitations

The automated scope is limited to `face` and `license_plate`. OCR, screen, document, badge, general text, SSN, and other text-sensitive detection are not implemented. The current release is a local single-user workflow, not a hardened remote multi-tenant service. Authentication, authorization, encrypted transport, central audit logging, backup, retention enforcement, and privacy-governance review belong to the deployment layer. The system does not make a blanket zero-leak guarantee; Final depends on compatible evidence, validated rendering, and full-video human approval.

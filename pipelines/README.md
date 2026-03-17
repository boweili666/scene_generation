# Pipelines

This directory is the normalized home for pipeline documentation.

Current executable pipeline code remains in legacy locations to avoid breaking runtime behavior:

- Real2Sim scripts: `option2_pipeline/`
- Isaac scene scripts: `isaac_local/scripts/`

The runtime and tests have already been moved to the new `runtime/` and `logs/` layout.
Future cleanup can physically relocate the legacy scripts after external callers are updated.

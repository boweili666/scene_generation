# Runtime Layout

The project now keeps mutable outputs out of source directories.

## Write Locations

- `runtime/uploads`: uploaded input images
- `runtime/renders`: rendered preview PNGs
- `runtime/scene_graph`: current scene graph JSON
- `runtime/real2sim`: masks, meshes, GLBs, poses, intermediate pipeline outputs
- `runtime/scene_service/usd`: generated USD scene files
- `logs`: runtime log files

## Compatibility

- Legacy `server/*.py` entrypoints still work.
- `server/config.py` and related modules forward to `app/backend/*`.
- `scripts/migrate_runtime_layout.py` can seed the new layout from old paths.

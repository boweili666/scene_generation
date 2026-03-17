# Runtime Layout

The project now keeps mutable outputs out of source directories.

## Write Locations

- `runtime/uploads`: uploaded input images
- `runtime/renders`: rendered preview PNGs
- `runtime/scene_graph`: current scene graph JSON
- `runtime/real2sim`: masks, meshes, GLBs, poses, intermediate pipeline outputs
- `runtime/scene_service/usd`: generated USD scene files
- `runtime/scene_service/placements`: persisted scene-service placements
- `logs`: runtime log files

## Entry Points

- Web app: `python -m app.backend.app`
- Scene service: `python -m app.backend.services.scene_service`

# Architecture

## Layout

- `app/backend`: active Flask/FastAPI backend code
- `app/frontend`: active frontend shell, CSS, JS, vendor assets
- `runtime`: mutable runtime artifacts
- `logs`: mutable log files
- `pipelines/real2sim`: active Real2Sim pipeline scripts
- `pipelines/isaac`: Isaac helper scripts and static USD assets

## Backend

- `app/backend/app.py`: Flask app factory and static serving
- `app/backend/api/routes.py`: HTTP routes and job orchestration
- `app/backend/services/openai_service.py`: scene graph and JSON editing OpenAI calls
- `app/backend/services/instruction_router.py`: rule-based instruction signals and backend route validation
- `app/backend/services/instruction_service.py`: unified `/apply_instruction` orchestration across scene graph and placements
- `app/backend/services/pipeline_service.py`: Real2Sim step execution and log streaming
- `app/backend/services/scene_service.py`: long-running Isaac FastAPI service
- `app/backend/config/settings.py`: runtime paths, service endpoints, environment overrides

## Frontend

- `app/frontend/index.html`: page shell
- `app/frontend/assets/css/*.css`: split stylesheets
- `app/frontend/assets/js/state.js`: shared globals and Three.js viewer primitives
- `app/frontend/assets/js/ui.js`: UI helpers, progress, upload preview, diagnostics widgets
- `app/frontend/assets/js/graph.js`: scene-graph fetch/render flow plus direct node/edge editing and save-back
- `app/frontend/assets/js/sim.js`: Real2Sim polling and scene service calls
- `app/frontend/assets/js/model.js`: unified instruction workflow for graph + placement edits
- `app/frontend/assets/js/boot.js`: event wiring and initial boot

## Runtime Data

- `runtime/uploads/latest_input.jpg`: latest uploaded reference image
- `runtime/renders/render.png`: latest scene-service render
- `runtime/scene_graph/current_scene_graph.json`: active scene graph
- `runtime/real2sim/*`: masks, meshes, GLBs, poses, intermediate artifacts
- `runtime/scene_service/usd/*`: generated USD outputs
- `runtime/scene_service/placements/placements_default.json`: persisted placements snapshot
- `logs/real2sim.log`: Real2Sim streaming log
- `logs/scene_service.log`: scene service process log

## Real2Sim Flow

Current active Real2Sim flow:

1. `pipelines/real2sim/object_segmentation_pipeline.py`
2. `pipelines/real2sim/streaming_generation_client.py`

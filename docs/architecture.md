# Architecture

## Layout

- `app/backend`: active Flask/FastAPI backend code
- `app/frontend`: active frontend shell, CSS, JS, vendor assets
- `runtime`: mutable runtime artifacts
- `logs`: mutable log files
- `server`: compatibility wrappers for old entrypoints
- `option2_pipeline`: legacy Real2Sim pipeline scripts
- `isaac_local`: legacy Isaac scene building scripts

## Backend

- `app/backend/app.py`: Flask app factory and static serving
- `app/backend/api/routes.py`: HTTP routes and job orchestration
- `app/backend/services/openai_service.py`: scene graph and JSON editing OpenAI calls
- `app/backend/services/pipeline_service.py`: Real2Sim step execution and log streaming
- `app/backend/services/scene_service.py`: long-running Isaac FastAPI service
- `app/backend/config/settings.py`: runtime paths, service endpoints, environment overrides

## Frontend

- `app/frontend/index.html`: page shell
- `app/frontend/assets/css/*.css`: split stylesheets
- `app/frontend/assets/js/state.js`: shared globals and Three.js viewer primitives
- `app/frontend/assets/js/ui.js`: UI helpers, progress, upload preview, diagnostics widgets
- `app/frontend/assets/js/graph.js`: scene-graph fetch/render/generation flow
- `app/frontend/assets/js/sim.js`: Real2Sim polling and scene service calls
- `app/frontend/assets/js/model.js`: JSON editing workflow
- `app/frontend/assets/js/boot.js`: event wiring and initial boot

## Runtime Data

- `runtime/uploads/latest_input.jpg`: latest uploaded reference image
- `runtime/renders/render.png`: latest scene-service render
- `runtime/scene_graph/current_scene_graph.json`: active scene graph
- `runtime/real2sim/*`: masks, meshes, GLBs, poses, intermediate artifacts
- `runtime/scene_service/usd/*`: generated USD outputs
- `logs/real2sim.log`: Real2Sim streaming log
- `logs/scene_service.log`: scene service process log

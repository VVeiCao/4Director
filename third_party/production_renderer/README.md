# Production control renderer

`src/baseline_eval/adapters/v5_rotation/controls.py` is the depth-control
renderer used for the paper results, SHA-256
`c90b27a674017ba0940aa0c576f7b586c10621674386b4f52ef4c8f55b847283`. The adjacent
`baseline_eval` modules are the minimum it imports; their inner module names are
retained so those imports resolve. Its PyTorch3D camera conversion and
point-cloud depth rasterization are written for this release and render the same
depth, bit for bit, as the renderer the paper used.

Stage 3 runs it for every Case. For the three teaser Cases it renders the scene
and meshes in `data/demos/hiker-camel`, and stage 4 then reproduces the output
SHA-256 recorded under `hashes.output` in each `configs/cases/*.yaml`.

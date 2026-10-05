# Third-party and vendored source

Upstream projects are Git submodules pinned to one commit, so the checkout
itself stays clean:

- `DiffSynth-Studio` — the Wan video pipeline. Stage 4 imports it directly,
  unpatched.
- `Pixal3D` — single-image mesh reconstruction. Stage 2 runs it through
  `pixal3d_adapter/run_patched.py`, which applies `patches/pixal3d.tracked.patch`
  to a throwaway copy of the pinned commit. The patch loads the background
  remover lazily; unpatched, Pixal3D fetches the gated `briaai/RMBG-2.0` at
  start-up even though every Case passes an RGBA crop.
- `TAPIP3D` — carries the MegaSAM (camera tracking, UniDepth, RAFT, CVD) that
  stage 1 and the app's scene geometry run, with `megasam_final.pth`.
  `environments/install_extensions.sh` copies it into the environment and
  applies `patches/tapip3d.tracked.patch`, which makes MegaSAM's depth
  alignment ignore invalid pixels.

The rest is 4Director code that each stage runs as its own subprocess of the
same environment:

- `alignment/align_mesh.py` — stage 2. A Pixal3D mesh arrives in its own
  arbitrary frame, so this solves a robust Sim(3) from the mesh-to-pixel
  correspondences and the camera contract, then places the mesh in scene
  coordinates. It is fail-closed: median, p90, and p95 reprojection error and a
  minimum unique-pair count must all pass, otherwise it raises instead of
  emitting a misaligned mesh.
- `inference/wan_vace/run_inference.py` — stage 4. Loads the Wan2.1-VACE-14B
  base plus the step-2610 Motion Adapter and generates the video from the
  reference image, the control video, the prompt, and the seed.
- `production_renderer/` — the depth-control renderer used for the paper
  results. Its inner module names are retained from the original implementation
  so its imports continue to resolve.
- `scripts/evaluation/vendor/pixal3d/` — stage 2's helpers around Pixal3D: one
  generates a mesh with its pixel correspondences, the other solves the
  similarity that aligns it. The directory path doubles as their import path.

Nothing here writes into the repository. Every command takes an output
directory under `output/`.

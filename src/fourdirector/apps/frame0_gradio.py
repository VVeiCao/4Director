"""Single-image upload, reconstruction, motion, and preview UI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fourdirector.constants import FRAME_HEIGHT, FRAME_WIDTH

# Loopback only, and never a public Gradio tunnel: this page drives local GPUs.
DEFAULT_HOST = "localhost"
DEFAULT_GRADIO_SHARE = False
# Every event that reads or changes a run waits in this one queue. Gradio gives
# each listener its own queue by default, which would let a mask and a 3D job
# share the GPU, or two motion buttons write the same run at once.
RUN_QUEUE = "frame0-run"
PRESET_BUTTONS = (
    ("object-rotate-90", "Turn object 90°"),
    ("object-rotate-180", "Turn object 180°"),
    ("object-move-right", "Move object right"),
    ("object-move-left", "Move object left"),
    ("camera-yaw-p120", "Orbit camera +120°"),
    ("camera-yaw-n120", "Orbit camera −120°"),
)
# Each step is a card with a large numbered title, so the page reads as six
# separate steps rather than one long form.
STEP_CSS = """
.step-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  padding: 18px 20px !important;
  background: var(--background-fill-secondary);
}
/* Cards in one row share a height; the spare space stays below a card's last control. */
.step-card > * {
  flex-grow: 0 !important;
}
.step-heading {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 4px;
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.25;
  color: var(--body-text-color);
}
.step-number {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 50%;
  background: var(--color-accent);
  color: #fff;
  font-size: 1.15rem;
}
"""


def step_heading(number: int, title: str, note: str = "") -> None:
    """The large numbered title of one step, and an optional line under it."""
    import gradio as gr

    gr.HTML(
        f'<div class="step-heading"><span class="step-number">{number}</span>'
        f"<span>{title}</span></div>",
        padding=False,
    )
    if note:
        gr.Markdown(note)


def click_overlay(image: Any, clicks: list[dict[str, Any]], mask: Any = None) -> Any:
    """Render mask/click feedback without mutating the uploaded image."""
    import numpy as np
    from PIL import Image, ImageDraw

    base = Image.open(image).convert("RGBA") if isinstance(image, (str, Path)) else Image.fromarray(
        np.asarray(image).astype(np.uint8)
    ).convert("RGBA")
    if mask:
        mask_image = Image.open(mask).convert("L").resize(base.size, Image.Resampling.NEAREST)
        tint = Image.new("RGBA", base.size, (30, 210, 110, 0))
        tint.putalpha(mask_image.point(lambda value: 90 if value else 0))
        base = Image.alpha_composite(base, tint)
    draw = ImageDraw.Draw(base)
    radius = max(4, round(min(base.size) / 80))
    for click in clicks:
        x, y = int(click["x"]), int(click["y"])
        color = (30, 230, 80, 255) if int(click["label"]) else (245, 55, 55, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.ellipse(
            (x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1),
            outline=(255, 255, 255, 255),
            width=2,
        )
    return np.asarray(base.convert("RGB"))


def configuration_report(env_file: Any = None) -> str:
    """Name what is missing at startup, where an operator is looking."""
    from fourdirector.examples.frame0_components import missing_requirements
    from fourdirector.paths import COMPONENT_ENV_FILE

    missing = missing_requirements()
    source = f"Component paths loaded from {env_file}." if env_file else ""
    if not missing:
        return source or "All component paths are configured."
    lines = [
        f"  {step}: " + ", ".join(names) for step, names in missing.items()
    ]
    return "\n".join(
        part
        for part in (
            source,
            "These steps cannot run yet:",
            *lines,
            "Run download_models.py and environments/install_extensions.sh, or name "
            f"a file that lives elsewhere in ./{COMPONENT_ENV_FILE}.",
        )
        if part
    )


def _commands(workflow: Any, run_id: str) -> tuple[str, str]:
    def resolve(method: Any) -> str:
        try:
            return str(method(run_id))
        except Exception as exc:
            return f"# Unavailable until configured: {exc}"

    return resolve(workflow.render_command), resolve(workflow.generate_command)


def run_choices(workflow: Any, limit: int = 8) -> list[str]:
    """Label the most recent runs so a finished reconstruction is easy to spot."""
    labels = []
    for record in workflow.list_runs()[:limit]:
        state = "3D ready" if record["has_mesh"] else record["status"]
        preset = f" · {record['preset']}" if record.get("preset") else ""
        labels.append(f"{record['run_id']} · {state}{preset}")
    return labels


def run_id_from_choice(choice: Any) -> str:
    return str(choice or "").split(" · ")[0].strip()


def scene_preview(workflow: Any, run_id: str, tag: str) -> Any:
    """The GLB that shows the background, motion ghosts, and paths."""
    from fourdirector.examples.frame0_preview import build_preview, is_current

    root = Path(workflow.receipt(run_id)["root"])
    output = root / "preview" / f"{tag}.glb"
    if not is_current(root, output):
        build_preview(root, output)
    return str(output)


def motion_preview(workflow: Any, run_id: str, tag: str, view: str = "external") -> Any:
    """Animate the point cloud, either from outside or through the camera."""
    from fourdirector.examples.frame0_preview import build_motion_video, is_current

    root = Path(workflow.receipt(run_id)["root"])
    output = root / "preview" / f"{tag}-{view}.mp4"
    if not is_current(root, output):
        build_motion_video(root, output, view=view)
    return str(output)


def previews(workflow: Any, run_id: str, tag: str, *, animated: bool) -> tuple[Any, Any, Any]:
    """Return the 3D scene and the two clips, in the order the page shows them.

    The clips need an authored motion, so a freshly reconstructed run has the
    scene only. A preview newer than everything it is drawn from is reused, so
    reopening a run costs nothing when its motion has not changed.
    """
    return (
        scene_preview(workflow, run_id, tag),
        motion_preview(workflow, run_id, tag, "external") if animated else None,
        motion_preview(workflow, run_id, tag, "camera") if animated else None,
    )


NO_COMMANDS = ("# Generate 3D first", "# Generate 3D first")


class Frame0Callbacks:
    """What each control does, kept apart from how the page is laid out.

    Every method returns the outputs in the order its control declares them.
    """

    def __init__(self, workflow: Any, errors: Any) -> None:
        self.workflow = workflow
        # gradio.Error, injected so this class stays importable without gradio.
        self.errors = errors

    def _require_run(self, run_id: Any) -> str:
        """Refuse a run action on a page with no run, instead of looking up "None"."""
        if not run_id:
            raise self.errors("Create a run first.")
        return str(run_id)

    def _run_view(self, receipt: dict[str, Any]) -> tuple[Any, ...]:
        """Everything the page shows for one stored run, in build_app's run_view order."""
        run_id = receipt["run_id"]
        reconstructed = bool(receipt.get("mesh"))
        stage3, stage4 = _commands(self.workflow, run_id) if reconstructed else NO_COMMANDS
        return (
            run_id,
            receipt["image"],
            click_overlay(receipt["image"], receipt.get("clicks", []), receipt.get("mask")),
            receipt.get("mask"),
            receipt.get("mesh"),
            *(
                # Named after the run's motion, so the clips built when it was
                # chosen are the ones reused here.
                previews(
                    self.workflow,
                    run_id,
                    receipt.get("preset") or "reconstructed",
                    animated=bool(receipt.get("preset")),
                )
                if reconstructed
                else (None, None, None)
            ),
            receipt.get("prompt", ""),
            bool(receipt.get("mask_confirmed")),
            stage3,
            stage4,
        )

    def load_page(self) -> tuple[Any, ...]:
        """Show the most recently active run, read again on every page load.

        Reading it at load rather than at build keeps a refreshed page current
        and keeps a broken run from stopping the server from starting.
        """
        active = self.workflow.active_run_receipt()
        if not active:
            example = self.workflow.example_image
            return (
                None,
                str(example) if example is not None else None,
                None, None, None, None, None, None,
                "",
                False,
                *NO_COMMANDS,
                self.run_list(),
                "Create a run to begin.",
            )
        return (
            *self._run_view(active),
            self.run_list(),
            f"Active run: `{active['run_id']}`",
        )

    def check_upload(self, upload: Any) -> Any:
        """Refuse an upload that is not 832x480 as soon as it lands."""
        import gradio as gr
        from PIL import Image

        from fourdirector.examples.frame0_demo import upload_size_error

        if upload is None:
            return gr.update()
        with Image.open(upload) as image:
            error = upload_size_error(image.size)
        if error is None:
            return gr.update()
        gr.Warning(error, duration=None)
        return None

    def create_run(self, upload: Any) -> tuple[Any, ...]:
        if upload is None:
            raise self.errors("Choose an image first.")
        try:
            receipt = self.workflow.create_run(upload)
        except ValueError as error:
            raise self.errors(str(error)) from error
        return (
            receipt["run_id"],
            click_overlay(receipt["image"], []),
            None, None, None, None, None,
            "",
            False,
            *NO_COMMANDS,
            self.run_list(),
            f"Created run `{receipt['run_id']}`. Add positive and optional negative clicks.",
        )

    def restore_run(self, choice: Any) -> tuple[Any, ...]:
        run_id = run_id_from_choice(choice)
        if not run_id:
            raise self.errors("Pick a run to restore.")
        receipt = self.workflow.activate_run(run_id)
        return (
            *self._run_view(receipt),
            f"Restored `{run_id}` ({receipt.get('status')}).",
        )

    def add_click(self, label: str, run_id: str, event: Any) -> tuple[Any, bool, str]:
        run_id = self._require_run(run_id)
        index = getattr(event, "index", None)
        if not isinstance(index, (tuple, list)) or len(index) < 2:
            raise self.errors("The click did not include image coordinates.")
        receipt = self.workflow.add_click(
            run_id, int(index[0]), int(index[1]), 1 if label == "Positive" else 0
        )
        return (
            click_overlay(receipt["image"], receipt["clicks"]),
            False,
            f"{len(receipt['clicks'])} click(s); mask confirmation reset.",
        )

    def undo(self, run_id: str) -> tuple[Any, bool, str]:
        receipt = self.workflow.undo_click(self._require_run(run_id))
        return click_overlay(receipt["image"], receipt["clicks"]), False, "Last click removed."

    def clear(self, run_id: str) -> tuple[Any, Any, bool, str]:
        receipt = self.workflow.clear_clicks(self._require_run(run_id))
        return click_overlay(receipt["image"], []), None, False, "Clicks and mask cleared."

    def generate_mask(self, run_id: str) -> tuple[Any, Any, bool, str]:
        receipt = self.workflow.generate_mask(self._require_run(run_id))
        return (
            click_overlay(receipt["image"], receipt["clicks"], receipt["mask"]),
            receipt["mask"],
            False,
            "Mask generated. Inspect it, then explicitly confirm it.",
        )

    def confirm_mask(self, run_id: str) -> tuple[bool, str]:
        self.workflow.confirm_mask(self._require_run(run_id))
        return True, "Mask explicitly confirmed for 3D generation."

    def generate_3d(self, run_id: str, confirmed: bool) -> tuple[Any, ...]:
        run_id = self._require_run(run_id)
        if not confirmed:
            raise self.errors("Explicitly confirm the generated mask first.")
        receipt = self.workflow.generate_3d(run_id)
        stage3, stage4 = _commands(self.workflow, run_id)
        return (
            receipt["mesh"],
            *previews(self.workflow, run_id, "reconstructed", animated=False),
            receipt["prompt"],
            stage3,
            stage4,
            self.run_list(),
            f"3D ready for `{run_id}`. Pick a motion below.",
        )

    def save_caption(self, run_id: str, prompt: str) -> str:
        self.workflow.set_prompt(self._require_run(run_id), prompt)
        return "Caption saved to the run configuration."

    def choose_preset(self, run_id: str, preset: str) -> tuple[Any, ...]:
        from fourdirector.examples.frame0_presets import apply_frame0_preset

        # Check this page's run, not the active-run pointer another tab may have moved.
        if not run_id or not self.workflow.receipt(run_id).get("mesh"):
            raise self.errors("Generate 3D before choosing a motion.")
        receipt = self.workflow.apply_preset(run_id, preset, builder=apply_frame0_preset)
        stage3, stage4 = _commands(self.workflow, run_id)
        return (
            *previews(self.workflow, run_id, preset, animated=True),
            stage3,
            stage4,
            self.run_list(),
            f"**{dict(PRESET_BUTTONS)[preset]}** applied to `{receipt['run_id']}`. "
            "The clips show the new motion from outside and through the camera.",
        )

    def run_list(self) -> Any:
        import gradio as gr

        return gr.update(choices=run_choices(self.workflow))


def build_app(workflow: Any) -> Any:
    import gradio as gr

    handlers = Frame0Callbacks(workflow, gr.Error)
    example = str(workflow.example_image) if workflow.example_image is not None else None

    def add_click(label: str, run_id: str, event: Any) -> tuple[Any, bool, str]:
        return handlers.add_click(label, run_id, event)

    # Gradio passes the click position only to a parameter annotated SelectData,
    # and this module's postponed annotations would leave that a string.
    add_click.__annotations__["event"] = gr.SelectData

    # The components start empty; load_page fills them from disk on every visit.
    # Gradio 5 takes the CSS here (6 moves it to launch; pyproject pins <6).
    with gr.Blocks(title="4Director — Frame-0", css=STEP_CSS) as application:
        run_state = gr.State(None)
        mask_confirmed = gr.State(False)
        gr.Markdown(
            "# 4Director\n"
            "Turn one photo into a video with the motion you choose. Work down the "
            "page: pick an image, click the subject you want to move, then choose how "
            "it or the camera should travel. A golden retriever photo is loaded as an example."
        )
        status = gr.Markdown("Create a run to begin.")
        with gr.Row():
            # A radio keeps every stored run visible; a dropdown at the top of the
            # page opens upward and covers its own label.
            run_picker = gr.Radio(
                choices=run_choices(workflow),
                value=None,
                label="Reopen a finished run, with its mask, mesh, and caption",
                scale=4,
            )
            restore_button = gr.Button("Reopen", scale=1)

        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="step-card"):
                step_heading(
                    1,
                    "Upload an image",
                    f"It must be exactly {FRAME_WIDTH}×{FRAME_HEIGHT} pixels, the size every "
                    "stage works at; other sizes are refused rather than stretched.",
                )
                upload = gr.Image(
                    value=example,
                    label=f"Upload a {FRAME_WIDTH}×{FRAME_HEIGHT} image",
                    type="filepath",
                    height=360,
                )
                if example is not None:
                    gr.Examples(
                        examples=[[example]],
                        inputs=[upload],
                        label="Example: frame 0 of the DAVIS 2017 dog sequence, resized to 832×480",
                    )
                create = gr.Button("Create run", variant="primary")
            with gr.Column(elem_classes="step-card"):
                step_heading(2, "Click the object")
                click_mode = gr.Radio(
                    ("Positive", "Negative"), value="Positive", label="Click type"
                )
                annotated = gr.Image(
                    label="Object selection",
                    type="numpy",
                    interactive=False,
                    height=360,
                )
                with gr.Row():
                    undo_button = gr.Button("Undo click")
                    clear_button = gr.Button("Clear")

        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="step-card"):
                step_heading(3, "Generate and confirm the mask")
                mask_button = gr.Button("Generate mask")
                mask_preview = gr.Image(
                    label="Generated mask",
                    type="filepath",
                    height=360,
                )
                confirm_button = gr.Button("Confirm mask")
            with gr.Column(elem_classes="step-card"):
                step_heading(4, "Reconstruct in 3D")
                generate_button = gr.Button("Generate 3D", variant="primary")
                mesh = gr.Model3D(
                    label="Object mesh",
                    clear_color=(1.0, 1.0, 1.0, 1.0),
                    height=360,
                )
                caption = gr.Textbox(label="Auto-caption (editable)", lines=3)
                caption_button = gr.Button("Save caption")

        with gr.Column(elem_classes="step-card"):
            step_heading(
                5,
                "Choose how it moves",
                "Each button rewrites the 81-frame motion. Drag and zoom the 3D scene to "
                "inspect the two paths, then watch the clips: one from outside, showing the "
                "object and the camera travel, one through the camera itself.",
            )
            preset_buttons = []
            with gr.Row(equal_height=True):
                for name, label in PRESET_BUTTONS:
                    preset_buttons.append((name, gr.Button(label, min_width=100)))
            preview = gr.Model3D(
                label="Frame 0 in 3D: green is the object path, blue is the camera path",
                clear_color=(1.0, 1.0, 1.0, 1.0),
                height=560,
            )
            with gr.Row():
                motion = gr.Video(
                    label="From outside: object and camera, always fully in frame",
                    autoplay=True,
                    loop=True,
                    height=360,
                )
                shot = gr.Video(
                    label="Camera view: the framing the generated video will have",
                    autoplay=True,
                    loop=True,
                    height=360,
                )

        with gr.Column(elem_classes="step-card"):
            step_heading(
                6,
                "Render and generate",
                "Run these two commands in a terminal: the first renders the depth control "
                "for the chosen motion, the second generates the video. To try another seed, "
                "rerun only the second with another --seed.",
            )
            stage3 = gr.Code(value=NO_COMMANDS[0], language="shell", label="Stage 3 command")
            stage4 = gr.Code(value=NO_COMMANDS[1], language="shell", label="Stage 4 command")

        run_view = [
            run_state,
            upload,
            annotated,
            mask_preview,
            mesh,
            preview,
            motion,
            shot,
            caption,
            mask_confirmed,
            stage3,
            stage4,
        ]
        queued = {"concurrency_id": RUN_QUEUE}
        application.load(handlers.load_page, None, [*run_view, run_picker, status], **queued)
        upload.upload(handlers.check_upload, upload, upload)
        create.click(
            handlers.create_run,
            upload,
            [
                run_state,
                annotated,
                mask_preview,
                mesh,
                preview,
                motion,
                shot,
                caption,
                mask_confirmed,
                stage3,
                stage4,
                run_picker,
                status,
            ],
            **queued,
        )
        restore_button.click(
            handlers.restore_run, run_picker, [*run_view, status], **queued
        )
        annotated.select(
            add_click,
            [click_mode, run_state],
            [annotated, mask_confirmed, status],
            **queued,
        )
        undo_button.click(
            handlers.undo, run_state, [annotated, mask_confirmed, status], **queued
        )
        clear_button.click(
            handlers.clear,
            run_state,
            [annotated, mask_preview, mask_confirmed, status],
            **queued,
        )
        mask_button.click(
            handlers.generate_mask,
            run_state,
            [annotated, mask_preview, mask_confirmed, status],
            **queued,
        )
        confirm_button.click(
            handlers.confirm_mask, run_state, [mask_confirmed, status], **queued
        )
        generate_button.click(
            handlers.generate_3d,
            [run_state, mask_confirmed],
            [mesh, preview, motion, shot, caption, stage3, stage4, run_picker, status],
            **queued,
        )
        caption_button.click(
            handlers.save_caption, [run_state, caption], status, **queued
        )
        for name, button in preset_buttons:
            button.click(
                handlers.choose_preset,
                [run_state, gr.State(name)],
                [preview, motion, shot, stage3, stage4, run_picker, status],
                **queued,
            )
    return application.queue(default_concurrency_limit=1)


def launch(
    *,
    host: str = DEFAULT_HOST,
    port: int = 7860,
    prevent_thread_lock: bool = False,
    quiet: bool = False,
) -> Any:
    # One server, one port: everything the page shows is rendered by this process.
    from fourdirector.examples.frame0_demo import (
        create_workflow,
        load_component_environment,
    )
    from fourdirector.paths import use_checkout_caches

    env_file = load_component_environment()
    use_checkout_caches()
    if not quiet:
        print(configuration_report(env_file), flush=True)
    workflow = create_workflow()
    application = build_app(workflow)
    return application.launch(
        server_name=host,
        server_port=int(port),
        share=DEFAULT_GRADIO_SHARE,
        allowed_paths=list(workflow.gradio_allowed_paths),
        prevent_thread_lock=prevent_thread_lock,
        quiet=quiet,
        show_error=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

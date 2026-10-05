#!/usr/bin/env python3
"""Generate one editable video prompt from an uploaded reference image."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

DEFAULT_REQUEST = (
    "Write one concise English video-generation caption for this image. "
    "Describe the visible subject, scene, lighting, camera realism, and plausible "
    "motion. Do not mention masks, controls, rendering, prompts, or metadata."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Hugging Face revision of --model (default: its main branch)")
    parser.add_argument("--request", default=DEFAULT_REQUEST)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(args.image)},
            {"type": "text", "text": str(args.request)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    # Qwen3-VL samples (its generation_config sets do_sample), so the seed is
    # what makes the same image get the same caption on the same GPU and software.
    set_seed(args.seed)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens))
    trimmed = [
        output[len(source) :]
        for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    caption = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    caption = " ".join(caption.strip().split())
    if not caption:
        raise RuntimeError("Qwen3-VL returned empty text")
    atomic_json(
        args.output_json,
        {"text": caption, "source": "auto_caption", "model_id": str(args.model), "seed": args.seed},
    )
    print(caption, flush=True)


if __name__ == "__main__":
    main()

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a top-down (bird's-eye) view from a scene image with OpenAI image editing."
    )
    parser.add_argument(
        "--input",
        default="../sam3/scene_60.jpeg",
        help="Path to the input scene image.",
    )
    parser.add_argument(
        "--output",
        default="option2_pipeline/runtime/scene_60_top_view.png",
        help="Path for the generated output image.",
    )
    parser.add_argument(
        "--model",
        default="gpt-image-1.5",
        help="Image model name (for example: gpt-image-1, gpt-image-1.5).",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=["low", "medium", "high", "auto"],
        help="Output quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()

    prompt = (
        "Use the input image as the scene reference. "
        "Generate a clean, realistic top-down (bird's-eye, 90-degree overhead) view "
        "of the same scene layout. "
        "Do not rearrange objects. "
        "Do not add or remove any object. "
        "Keep scene topology unchanged. "
        "Keep object identities and relative spatial relationships strictly consistent: "
        "left-right, front-back, distance/proximity, and adjacency between objects must remain unchanged."
    )

    with input_path.open("rb") as img_file:
        result = client.images.edit(
            model=args.model,
            image=img_file,
            prompt=prompt,
            input_fidelity="high",
            quality=args.quality,
            size="1536x1024",
        )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()

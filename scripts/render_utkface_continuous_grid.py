#!/usr/bin/env python3
"""Render a generated-only UTKFace grid under continuous age conditions.

Each row reuses the same latent random stream across age columns.  Therefore,
horizontal changes expose the effect of interpolating the age-conditioned
content prior more clearly than independently sampled columns would.  The
result is qualitative and is not an age-accuracy measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_face_contract_latents import _age_knots, _load_model  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--ages", type=float, nargs="+", default=[5, 15, 25, 35, 50, 65, 80, 100])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.rows < 1 or len(args.ages) < 2:
        raise ValueError("rows must be positive and at least two ages are required")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    model = _load_model(checkpoint, args.image_size, device)
    ages = torch.tensor(args.ages, dtype=torch.float32, device=device)
    # Sample each row once for all columns. Re-seeding the continuous sampler
    # gives identical epsilon/style draws while condition-dependent centers vary.
    row_images = []
    for row in range(args.rows):
        row_latents = []
        for age in ages:
            # Reset to the row seed for every column so both the spherical
            # epsilon and Gaussian style draw are exactly paired across ages.
            generator = torch.Generator(device="cpu").manual_seed(args.seed + row)
            content, style = model.sample_from_continuous_prior(
                age.reshape(1), _age_knots(device), generator=generator
            )
            row_latents.append(torch.cat([content, style], dim=1))
        row_images.append(model.decoder(torch.cat(row_latents, dim=0)).cpu())
    images = torch.cat(row_images, dim=0)
    padding = 2
    grid = make_grid(images, nrow=len(args.ages), padding=padding, pad_value=1.0)
    grid_image = to_pil_image(grid)
    header_height = 22
    annotated = Image.new("RGB", (grid_image.width, grid_image.height + header_height), "white")
    annotated.paste(grid_image, (0, header_height))
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for column, age in enumerate(args.ages):
        label = f"age {age:g}"
        box = draw.textbbox((0, 0), label, font=font)
        width = box[2] - box[0]
        center = padding + column * (args.image_size + padding) + args.image_size / 2
        draw.text((center - width / 2, 3), label, fill="black", font=font)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    annotated.save(temporary)
    temporary.replace(output)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "ages_by_column": [float(value) for value in args.ages],
        "rows": int(args.rows),
        "seed_by_row": [int(args.seed + row) for row in range(args.rows)],
        "sampling_note": (
            "Generated-only qualitative grid; each row reuses its random stream "
            "across age columns. It is not an age-retention estimator."
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

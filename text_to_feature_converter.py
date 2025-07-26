import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model


def load_model():
    """Build datamodule and model and load checkpoints."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]  # use defaults
    cfg = parse_args(phase="webui")
    sys.argv = saved

    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    if cfg.TEST.CHECKPOINTS:
        state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
    else:
        print("Warning: no checkpoints provided, using untrained model")
    device = torch.device("cuda" if cfg.ACCELERATOR == "gpu" else "cpu")
    model.to(device)
    model.eval()
    return model, datamodule, device


def text_to_motion(model, datamodule, device, texts):
    """Generate motion npy arrays from input texts.txt."""
    lengths = [datamodule.hparams.max_motion_length] * len(texts)
    batch = {"text": texts, "length": lengths}
    outputs = model(batch, task="t2m")
    motions = []
    for j, l in zip(outputs["feats"], outputs["length"]):
        motions.append(j[:l].detach().cpu().numpy())
    return motions


def main():
    with open("input_scripts/texts.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    output_path = Path("results/npy")

    model, datamodule, device = load_model()

    motions = text_to_motion(model, datamodule, device, texts)

    if len(motions) == 1 and not output_path.exists():
        np.save(output_path, motions[0])
        print(f"Motion saved to {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(motions):
            fname = output_path / f"{names[i]}.npy"
            np.save(fname, m)
            print(f"Motion saved to {fname}")


if __name__ == "__main__":
    main()
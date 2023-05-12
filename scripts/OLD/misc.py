import click
import logging
import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def collect_iq_shapes():
    """Collects the shapes of the IQ images"""
    print("Collecting IQ shapes")

    from tqdm import tqdm
    from src.data.exact.server import load_by_core_specifier
    from src.data.exact.resources import metadata

    table = metadata()

    all_info = []
    for i, row in tqdm(table.iterrows(), total=len(table)):
        iq = load_by_core_specifier(row.core_specifier)
        iq_shape = iq["Q"].shape
        all_info.append(
            {
                "core_specifier": row.core_specifier,
                "iq_axial_pixels": iq_shape[0],
                "iq_lateral_pixels": iq_shape[1],
                "iq_num_frames": iq_shape[2],
            }
        )

    all_info = pd.DataFrame(all_info)
    all_info.to_csv("iq_shapes.csv")


if __name__ == "__main__":
    collect_iq_shapes()

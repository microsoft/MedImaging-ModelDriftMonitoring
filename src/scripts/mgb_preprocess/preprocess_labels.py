from pathlib import Path
from typing import List

import click
import pandas as pd

from pycrumbs import tracked

from model_drift import mgb_locations
from model_drift.data import mgb_data


@click.command()
@click.argument(
    "labels-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=mgb_locations.labels_csv,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=mgb_locations.preprocessed_labels_dir,
)
@tracked(directory_parameter="output_dir")
def preprocess_labels(labels_csv: Path, output_dir: Path) -> None:
    """Preprocess labels for this project."""
    labels_df = pd.read_csv(
        labels_csv,
        index_col=0,
        dtype={
            'PatientID': str,
            'AccessionNumber': str,
            'StudyDate': str,
            'StudyInstanceUID': str
        }
    )

    non_label_cols = [
        c for c in labels_df.columns if c not in mgb_data.RAW_LABELS
    ]
    out_df = labels_df[non_label_cols].copy()

    # Map missing values to 0 and equivocal values (-1) to 1
    labels_df.fillna(0.0, inplace=True)
    for c in mgb_data.RAW_LABELS:
        labels_df[c] = labels_df[c].abs()

    # Apply label groupings
    def group_row(row: pd.Series, sub_labels: List[str]):
        return max(row[c] for c in sub_labels)

    for out_col, sub_labels in mgb_data.LABEL_GROUPINGS.items():
        out_df[out_col] = labels_df.apply(
            lambda row: group_row(row, sub_labels),
            axis=1
        )

    out_df.to_csv(output_dir / "labels.csv")


if __name__ == "__main__":
    preprocess_labels()

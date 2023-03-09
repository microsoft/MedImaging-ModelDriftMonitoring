from pathlib import Path
import click
import pandas as pd

from model_drift import mgb_locations


@click.command()
@click.argument(
    "raw-labels-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=mgb_locations.raw_labels_csv,
)
def merge_labels(raw_labels_csv: Path) -> None:
    """Merge labels to create an anonymized set."""
    labels_df = pd.read_csv(raw_labels_csv)
    study_df = pd.read_csv(
        mgb_locations.study_list_csv,
        index_col=0,
    )
    master_df = pd.concat([study_df, labels_df], axis=1)

    id_map_df = pd.read_csv(mgb_locations.crosswalk_csv, index_col=0)
    joined_df = master_df.merge(
        id_map_df,
        how="left",
        left_on="AccessionNumber",
        right_on="ORIG_AccNumber",
        validate="one_to_one",
    )
    out_df = (
        joined_df[
            ["ANON_MRN", "ANON_AccNumber", "ANON_SUID", "ORIG_StudyDate"]
            + list(labels_df.columns)
        ]
        .copy()
        .rename(
            columns={
                "ANON_MRN": "PatientID",
                "ANON_AccNumber": "AccessionNumber",
                "ANON_SUID": "StudyInstanceUID",
                "ORIG_StudyDate": "StudyDate",
            }
        )
    )
    out_df.to_csv(mgb_locations.labels_csv)


if __name__ == "__main__":
    merge_labels()

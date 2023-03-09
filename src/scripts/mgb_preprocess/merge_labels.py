import click
import pandas as pd

from cxr_drift import locations


@click.command()
def merge_labels() -> None:
    """Merge labels to create an anonymized set."""
    labels_df = pd.read_csv(locations.raw_labels_csv)
    study_df = pd.read_csv(
        locations.study_list_csv,
        index_col=0,
    )
    master_df = pd.concat([study_df, labels_df], axis=1)

    id_map_df = pd.read_csv(locations.id_map_csv, index_col=0)
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
    out_df.to_csv(locations.labels_csv)

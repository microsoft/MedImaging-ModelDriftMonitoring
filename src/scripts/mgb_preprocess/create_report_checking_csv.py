import click
import pandas as pd

from pycrumbs import tracked

from model_drift import mgb_locations


@click.command()
@click.argument("n", type=int)
@tracked(literal_directory=mgb_locations.report_checking_dir)
def create_report_checking_csv(n: int):
    """Create a CSV file of a random selection of reports to check.

    Sample is stratified by calendar month (and year).

    """
    # Load in raw impressions file
    impressions = pd.read_csv(
        mgb_locations.impressions_csv,
        header=None,
        names=["Impression"],
    )
    reports = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    reports = reports[["Patient MRN", "Accession Number", "Report Text"]].copy()
    reports = pd.concat([reports, impressions], axis=1)

    # Load in labels file
    labels = pd.read_csv(
        mgb_locations.labels_csv,
        dtype={"PatientID": str, "AccessionNumber": str, "StudyDate": str},
        index_col=0,
    )
    labels = labels[labels["AccessionNumber"].notnull()].copy()

    # Load in crosswalk table
    crosswalk = pd.read_csv(mgb_locations.crosswalk_csv, dtype=str)

    merged = (
        labels.merge(
            crosswalk,
            how="left",
            left_on="AccessionNumber",
            right_on="ANON_AccNumber",
            validate="one_to_one",
        )
        .merge(
            reports,
            how="left",
            left_on="ORIG_AccNumber",
            right_on="Accession Number",
            validate="one_to_one",
        )
    )
    merged["StudyDate"] = pd.to_datetime(merged.StudyDate)

    random_state = 12345
    sample = (
        merged.groupby(
            [
                merged.StudyDate.dt.year,
                merged.StudyDate.dt.month,
                merged.StudyDate.dt.day,
            ]
        )
        .sample(n=n, random_state=random_state)  # Sample n per group
        .sample(frac=1, random_state=random_state)  # Shuffle full result
    )
    sample["ImpressionExtractionError"] = ""
    sample = sample[
        [
            "Impression",
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            "ANON_MRN",
            "ANON_AccNumber",
            "Report Text",
            "ImpressionExtractionError",
        ]
    ].copy()

    sample.to_csv(
        mgb_locations.report_checking_dir / "reports_to_check.csv",
        index=False
    )


if __name__ == "__main__":
    create_report_checking_csv()

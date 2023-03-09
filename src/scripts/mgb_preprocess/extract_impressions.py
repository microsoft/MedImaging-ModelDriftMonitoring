from os import linesep

import click
import pandas as pd
from ml_proj_tools.records import auto_record

from cxr_drift import locations


def get_impression(report: str) -> str:
    """Get the impression section of a single report.

    Parameters
    ----------
    report: str
        The report string, raw.

    Returns
    -------
    str:
        The report's impression section.

    """
    start_markers = [
        "IMPRESSION:",
        "IMPRESSION.",
        "IMPRESSION",
        "FINDINGS:",
        "RECOMMENDATION:",
    ]
    for marker in start_markers:
        if marker in report:
            impression = report.split(marker)[1]
            break
    else:
        print("FAILED CASE:")
        print(report)
        print()
        impression = report

    end_markers = [
        "RECOMMENDATION:",
        "ATTESTATION:",
        "ATTESTATION.",
        "ATTESTATION",
    ]
    for marker in end_markers:
        if marker in report:
            impression = impression.split(marker)[0]
            break

    impression = " ".join(impression.splitlines()).strip() + linesep

    return impression


@click.command()
@auto_record(output_dir=locations.reports_dir)
def extract_impressions() -> None:
    """Extract impressions from the reports CSV and stored in impressions.csv."""
    reports_df = pd.read_csv(locations.master_reports_csv, dtype=str)
    impressions = reports_df["Report Text"].apply(get_impression)
    impressions_df = pd.DataFrame(
        {
            "impression": impressions,
        }
    )
    impressions_df.to_csv(
        locations.impressions_csv,
        header=False,
        index=False,
    )

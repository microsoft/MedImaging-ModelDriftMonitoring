from os import linesep
from pathlib import Path

import click
import pandas as pd
from pycrumbs import tracked

from model_drift import mgb_locations


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
@click.argument(
    "output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=mgb_locations.reports_dir,
)
@tracked(directory_parameter="output_dir")
def extract_impressions(output_dir: Path) -> None:
    """Extract impressions from the reports CSV and stored in impressions.csv."""
    reports_df = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    impressions = reports_df["Report Text"].apply(get_impression)
    impressions_df = pd.DataFrame(
        {
            "impression": impressions,
        }
    )
    impressions_df.to_csv(
        output_dir / "impressions.csv",
        header=False,
        index=False,
    )


if __name__ == "__main__":
    extract_impressions()

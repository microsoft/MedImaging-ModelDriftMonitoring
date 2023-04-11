from os import linesep
from pathlib import Path

import click
import pandas as pd
from pycrumbs import tracked

from model_drift import mgb_locations


def get_finding(report: str) -> str:
    """Get the impression and finding sections of a single report.

    Parameters
    ----------
    report: str
        The report string, raw.

    Returns
    -------
    str:
        The report's impression and finding sections.

    """
    start_markers = [
        "FINDINGS:",
        "IMPRESSION:",
        "IMPRESSION.",
        "IMPRESSION",
        "RECOMMENDATION:",
    ]
    for marker in start_markers:
        if marker in report:
            if len(report.split(marker)[1]) > 5:  # some reports have two FINDINGS titles.
                findings = report.split(marker)[1]
            else:
                findings = report.split(marker)[2]
            break
    else:
        print("FAILED CASE:")
        print(report)
        print()
        findings = report

    end_markers = [
        "RECOMMENDATION:",
        "ATTESTATION:",
        "ATTESTATION.",
        "ATTESTATION",
    ]
    for marker in end_markers:
        if marker in report:
            findings = findings.split(marker)[0]
            break

    # findings = " ".join(findings.splitlines()).strip() + linesep (version 0)

    # remove headings and retain line space ('\n') in findings
    ## the code is slow (~3 minutes)
    findings = " ".join(findings.splitlines()).strip()
    # check whether there is only one finding in the report
    # (e.g., there is only Impressions section and no Findings section)
    if len(findings.split(':')) != 1:
        findings = findings.split(':')[1:]
        for i, finding in enumerate(findings):
            # print(i, "Finding:\n", finding)
            # if we reach the last finding, add a line space at the end
            if i + 1 == len(findings):
                findings[i] = finding.strip() + linesep
                break
            # for all the other findings, remove the last word/phrase
            # (i.e., the original heading such as "Lungs" and "Heart and mediastinum")
            else:
                # if len(finding.split(". ")) != 1: (version 2)
                # findings[i] = '. '.join(finding.split('. ')[:-1]).strip() (version 1)
                finding = ' '.join(finding.split(' ')[:-1]).strip()
                # some headings have length > 1, we have to further remove some words/phrases
                if finding.endswith(' Heart and'):  # (i.e., Heart and mediastinum)
                    findings[i] = ' '.join(finding.split(' ')[:-2]).strip()
                elif finding.endswith(' Bones/Soft'):  # (i.e., Bones/Soft tissues)
                    findings[i] = ' '.join(finding.split(' ')[:-1]).strip()
                elif finding.endswith(' Bones and soft'):  # (i.e., Bones and soft tissues)
                    findings[i] = ' '.join(finding.split(' ')[:-3]).strip()
                else:
                    findings[i] = finding
        # join all the findings in a list back together
        # findings = '.\n'.join(findings) (version 1)
        findings = '\n'.join(findings)

    return findings


@click.command()
@click.argument(
    "output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=mgb_locations.reports_dir,
)
@tracked(directory_parameter="output_dir")
def extract_findings(output_dir: Path) -> None:
    """Extract impressions and findings from the reports CSV and stored in findings.csv."""
    reports_df = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    findings = reports_df["Report Text"].apply(get_finding)
    findings_df = pd.DataFrame(
        {
            "finding": findings,
        }
    )
    findings_df.to_csv(
        output_dir / "findings.csv",
        header=False,
        index=False,
    )


# split the findings dataframe into 10 smaller dataframes and store them
def split_findings():
    findings_df = pd.read_csv(mgb_locations.reports_dir / 'findings.csv', header=None)
    filename = 1
    filepath = "/Volumes/qtim/datasets/private/xray_drift/reports/"
    for i in range(findings_df.shape[0]):
        if i % 10000 == 0:
            df = findings_df.iloc[i:i+10000, :]
            df.to_csv(filepath + "findings" + str(filename) + ".csv", header=None)
            filename += 1


# combine and output the raw_labels.csv file (based on findings)
def output_raw_labels_df():
    # read in and combine all the raw_labels files
    raw_labels_df = pd.DataFrame()
    for i in range(10):
        filepath = str(mgb_locations.reports_dir) + "/raw_labels" + str(i + 1) + ".csv"
        raw_labels_df_tem = pd.read_csv(filepath)
        raw_labels_df_tem = raw_labels_df_tem.loc[(raw_labels_df_tem.Reports != '0'), :]
        raw_labels_df = pd.concat([raw_labels_df, raw_labels_df_tem], ignore_index=True)

    # output and store the combined raw labels file
    raw_labels_df.to_csv(mgb_locations.raw_labels_csv)


if __name__ == "__main__":
    # extract_findings()
    # split_findings()
    # output_raw_labels_df()

    # compare radiologist labeling with findings labels
    # select unique identifiers from the radiologist labeling file
    rad_labeling = pd.read_excel(mgb_locations.csv_dir / "radiologist_checked_labels.xlsx")
    rad_labeling = rad_labeling.loc[(rad_labeling.Impression.notna()), :]
    rad_labeling_unique_id = rad_labeling.loc[:, ['ANON_MRN', 'ANON_AccNumber']]

    # read in labels.csv and filter reports
    labels_df = pd.read_csv(mgb_locations.labels_csv)
    labels_df_subset = labels_df.merge(rad_labeling_unique_id,
                                       how="inner",
                                       left_on=['PatientID', 'AccessionNumber'],
                                       right_on=['ANON_MRN', 'ANON_AccNumber'])
    labels_df_subset.to_csv(mgb_locations.csv_dir / "labels_subset.csv")

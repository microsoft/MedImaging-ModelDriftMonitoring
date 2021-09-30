import pandas as pd


def prepare_padchest(df):
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    return df

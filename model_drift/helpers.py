import os
from pathlib import Path
import pandas as pd
from . import constants as C


def get_top_dir():
    return Path(os.path.dirname(os.path.realpath(__file__))).parent


def get_data_dir():
    return get_top_dir().joinpath(C.DATA_REL_TOP)


def get_data_dir():
    return get_top_dir().joinpath(C.DATA_REL_TOP)


def get_padchest_csv_filename():
    return get_data_dir().joinpath(C.PADCHEST_CSV_FILENAME)


def prepare_padchest(df):
    df['StudyDate'] = pd.to_datetime(df['StudyDate_DICOM'], format='%Y%m%d')
    df['PatientBirth'] = pd.to_datetime(df['PatientBirth'], format='%Y')
    return df

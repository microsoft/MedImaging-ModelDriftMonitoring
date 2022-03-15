#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy

import numpy as np
import pandas as pd

from .dataset import PadChestDataset
from .drift_data_base import ModelDriftData
from .utils import fix_strlst, split_on_date
from .. import settings

LABEL_MAP = {
    "Atelectasis": [
        "laminar atelectasis",
        "fibrotic band",
        "atelectasis",
        "lobar atelectasis",
        "segmental atelectasis",
        "atelectasis basal",
        "total atelectasis",
    ],
    "Cardiomegaly": ["cardiomegaly", "pericardial effusion"],
    "Consolidation": ["consolidation"],
    "Edema": ["kerley lines"],
    "Lesion": ["nodule", "pulmonary mass", "lung metastasis", "multiple nodules", "mass"],
    "No Finding": ["normal"],
    "Opacity": [
        "infiltrates",
        "alveolar pattern",
        "pneumonia",
        "interstitial pattern",
        "increased density",
        "consolidation",
        "bronchovascular markings",
        "pulmonary edema",
        "pulmonary fibrosis",
        "tuberculosis sequelae",
        "cavitation",
        "reticular interstitial pattern",
        "ground glass pattern",
        "atypical pneumonia",
        "post radiotherapy changes",
        "reticulonodular interstitial pattern",
        "tuberculosis",
        "miliary opacities",
    ],
    "Pleural Abnormalities": [
        "costophrenic angle blunting",
        "pleural effusion",
        "pleural thickening",
        "calcified pleural thickening",
        "calcified pleural plaques",
        "loculated pleural effusion",
        "loculated fissural effusion",
        "asbestosis signs",
        "hydropneumothorax",
        "pleural plaques",
    ],
    "Pleural Effusion": ["pleural effusion"],
    "Pneumonia": ["pneumonia"],
}

BAD_FILES = [
    "216840111366964012283393834152009026160348294_00-014-160.png",
    "216840111366964012283393834152009033102258826_00-059-087.png",
    "216840111366964012339356563862009041122518701_00-061-032.png",
    "216840111366964012339356563862009047085820744_00-054-000.png",
    "216840111366964012339356563862009068084200743_00-045-105.png",
    "216840111366964012339356563862009072111404053_00-043-192.png",
    "216840111366964012373310883942009111121552024_00-072-099.png",
    "216840111366964012373310883942009117084022290_00-064-025.png",
    "216840111366964012373310883942009170084120009_00-097-074.png",
    "216840111366964012373310883942009203115626970_00-031-135.png",
    "216840111366964012487858717522009251095944293_00-018-154.png",
    "216840111366964012558082906712009300162151055_00-078-079.png",
    "216840111366964012558082906712009327122220177_00-102-064.png",
    "216840111366964012558082906712009330202206556_00-102-040.png",
    "216840111366964012734950068292010154110220411_04-008-052.png",
    "216840111366964012734950068292010166125223829_04-006-138.png",
    "216840111366964012819207061112010306085429121_04-020-102.png",
    "216840111366964012819207061112010307142602253_04-014-084.png",
    "216840111366964012819207061112010314122154282_04-013-126.png",
    "216840111366964012819207061112010315104455352_04-024-184.png",
    "216840111366964012819207061112010320134721426_04-022-028.png",
    "216840111366964012819207061112010322100558680_04-001-153.png",
    "216840111366964012819207061112010322154706609_04-021-011.png",
    "216840111366964012904401302362010328092649206_04-014-085.png",
    "216840111366964012904401302362010329193325676_04-016-070.png",
    "216840111366964012904401302362010333080926354_04-019-163.png",
    "216840111366964012922382741642010350171324419_04-000-122.png",
    "216840111366964012922382741642011010093926179_00-126-037.png",
    "216840111366964012948363412702011018092612949_00-124-038.png",
    "216840111366964012959786098432011032091803456_00-172-113.png",
    "216840111366964012959786098432011033083840143_00-176-115.png",
    "216840111366964012959786098432011054135834306_00-176-162.png",
    "216840111366964012989926673512011068092304604_00-163-066.png",
    "216840111366964012989926673512011069111543722_00-165-111.png",
    "216840111366964012989926673512011074122523403_00-163-058.png",
    "216840111366964012989926673512011083122446341_00-158-003.png",
    "216840111366964012989926673512011101135816654_00-184-188.png",
    "216840111366964012989926673512011101154138555_00-191-086.png",
    "216840111366964012989926673512011132200139442_00-157-099.png",
    "216840111366964013076187734852011178154626671_00-145-086.png",
    "216840111366964013076187734852011259174838161_00-131-007.png",
    "216840111366964013076187734852011291090445391_00-196-188.png",
]


def read_padchest(csv_file=None) -> pd.DataFrame:
    csv_file = csv_file or settings.PADCHEST_FILENAME
    df = pd.read_csv(csv_file, low_memory=False, index_col=0)
    return prepare_padchest(df)


def prepare_padchest(df):
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    df["Labels"] = fix_strlst(df["Labels"])
    df["Frontal"] = df["Projection"].isin(["PA", "AP", "AP_horizontal"])
    df["age"] = (df["StudyDate"] - df["PatientBirth"]) / np.timedelta64(1, "Y")
    return df


class PadChest(ModelDriftData):
    LABEL_COL = "Labels"
    DATASET_COLS = [
        "ImageID",
        "ImageDir",
        "StudyDate",
        "Projection",
        "Frontal",
        "Labels",
    ]
    DATASET_CLASS = PadChestDataset

    IMAGEID_COL = "ImageID"

    def __init__(
            self,
            df,
            label_map=LABEL_MAP,
            bad_files=BAD_FILES,
    ) -> None:
        super().__init__(df, label_map=label_map)
        self.bad_files = bad_files

    def prepare(self):
        super().prepare()
        self.df = prepare_padchest(self.df)
        self.remove_bad_files()

    def remove_bad_files(self):
        if self.bad_files:
            self.df = self.df[~self.df[self.IMAGEID_COL].isin(self.bad_files)]

    def __copy__(self):
        return type(self)(df=copy.copy(self.df), label_map=copy.copy(self.label_map),
                          bad_files=copy.copy(self.bad_files))

    def split(self, split_dates, studydate_index=False):
        assert len(split_dates) > 0
        self.prepare()
        df = self.df
        if studydate_index:
            df = df.set_index("StudyDate", drop=False)

        for new_df in split_on_date(df, split_dates, col="StudyDate"):
            yield self._copy_with_df(new_df.copy())

    @staticmethod
    def read_csv(csv_file=None):
        csv_file = csv_file or settings.PADCHEST_FILENAME
        return pd.read_csv(csv_file, low_memory=False, index_col=0)

    @classmethod
    def splits(cls, csv_file=None, studydate_index=False, split_dates=None, **init_kwargs):
        split_dates = split_dates or settings.PADCHEST_SPLIT_DATES
        print(split_dates)
        assert len(split_dates) == 2
        parent = cls.from_csv(csv_file, **init_kwargs)
        return parent.split(split_dates, studydate_index=studydate_index)

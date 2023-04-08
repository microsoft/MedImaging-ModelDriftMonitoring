#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import re
from typing import List

import six

from model_drift.drift import ChiSqDriftCalculator, KSDriftCalculator, TabularDriftCalculator
from model_drift.drift import HistIntersectionCalculator, KdeHistPlotCalculator


def match_keys(keys, regex_list) -> List[str]:
    return list({key for key in keys for regex in regex_list if re.search(regex, key)})


def auto_generate_metric_calculator(dataframe, vae_cols=r"mu\..*", score_cols=r"activation\..*",
                                    metadata_cols=(r'.*_DICOM', r"^age", r"Exposure*"),
                                    exclude_cols=r"StudyDate_DICOM"):
    if isinstance(vae_cols, six.string_types):
        vae_cols = [vae_cols]

    if isinstance(score_cols, six.string_types):
        score_cols = [score_cols]

    if isinstance(metadata_cols, six.string_types):
        metadata_cols = [metadata_cols]

    if isinstance(exclude_cols, six.string_types):
        exclude_cols = [exclude_cols]

    exclude_cols = match_keys(list(dataframe), exclude_cols)
    dataframe = dataframe.drop(columns=exclude_cols)

    score_cols = match_keys(list(dataframe), score_cols)
    vae_cols = match_keys(list(dataframe), vae_cols)

    metadata_cols = match_keys(list(dataframe), metadata_cols)

    metadata_float_cols = [c for c in metadata_cols if dataframe[c].dtype.kind in "uif" and dataframe[c].nunique() > 50]
    metadata_cat_cols = [c for c in metadata_cols if c not in metadata_float_cols]

    def add_vae_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="appearance")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=100), drilldown=True, group="appearance")

    def add_score_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="ai")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="ai")

    def add_metadata_metrics(dwc: TabularDriftCalculator, col: str):
        if col in metadata_float_cols:
            dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="metadata")
        elif col in metadata_cat_cols:
            dwc.add_drift_stat(col, ChiSqDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, HistIntersectionCalculator(), drilldown=True, group="metadata")

    dwc = TabularDriftCalculator()

    for c in score_cols:
        add_score_metrics(dwc, c)

    for c in vae_cols:
        add_vae_metrics(dwc, c)

    for c in metadata_cols:
        add_metadata_metrics(dwc, c)

    return dwc


def padchest_default_config(dataframe, vae_cols=r"mu\..*", score_cols=r"activation\..*"):
    if isinstance(vae_cols, six.string_types):
        vae_cols = [vae_cols]

    if isinstance(score_cols, six.string_types):
        score_cols = [score_cols]

    metadata_float_cols = ["WindowCenter_DICOM", "WindowWidth_DICOM", "Rows_DICOM", "Columns_DICOM",
                           "ExposureInuAs_DICOM", "RelativeXRayExposure_DICOM"]
    metadata_cat_cols = ["Projection", "PatientSex_DICOM", "ViewPosition_DICOM", "Modality_DICOM", "Manufacturer_DICOM",
                         "PhotometricInterpretation_DICOM", "PixelRepresentation_DICOM",
                         "PixelAspectRatio_DICOM", "SpatialResolution_DICOM", "BitsStored_DICOM",
                         "XRayTubeCurrent_DICOM", "Exposure_DICOM"]
    metadata_age_cols = ["age"]

    def add_vae_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="appearance")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="appearance")

    def add_score_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="ai")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="ai")

    def add_metadata_metrics(dwc: TabularDriftCalculator, col: str):
        if col in metadata_age_cols:
            dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500, hist_tol=0, kde_tol=0), drilldown=True,
                               group="metadata")
        elif col in metadata_float_cols:
            dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="metadata")
        elif col in metadata_cat_cols:
            dwc.add_drift_stat(col, ChiSqDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, HistIntersectionCalculator(), drilldown=True, group="metadata")

    score_cols = match_keys(list(dataframe), score_cols)
    vae_cols = match_keys(list(dataframe), vae_cols)
    metadata_cols = metadata_age_cols + metadata_cat_cols + metadata_float_cols

    dwc = TabularDriftCalculator()
    for col in dataframe.columns:
        if col in score_cols:
            add_score_metrics(dwc, col)
        elif col in vae_cols:
            add_vae_metrics(dwc, col)
        elif col in metadata_cols:
            add_metadata_metrics(dwc, col)
    return dwc


def mgb_default_config(dataframe, vae_cols=r"mu\..*", score_cols=r"activation\..*"):

    if isinstance(vae_cols, six.string_types):
        vae_cols = [vae_cols]

    if isinstance(score_cols, six.string_types):
        score_cols = [score_cols]

    metadata_float_cols = [
        "WindowCenter",
        "WindowWidth",
        "RelativeXRayExposure",
        "Rows",
        "Columns",
        "XRayTubeCurrent",
        "Exposure",
        "ExposureInuAs",
        "KVP",
    ]

    metadata_cat_cols = [
        "ViewPosition",
        "Manufacturer",
        "PhotometricInterpretation",
        "BitsStored",
        "Modality",
        "PixelRepresentation",
        "PixelAspectRatio",
        "SpatialResolution",
        "Point of Care",
        "Patient Sex",
        "Is Stat",
        "Exam Code",
    ]

    metadata_age_cols = ["Patient Age"]

    def add_vae_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="appearance")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="appearance")

    def add_score_metrics(dwc: TabularDriftCalculator, col: str):
        dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="ai")
        dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="ai")

    def add_metadata_metrics(dwc: TabularDriftCalculator, col: str):
        if col in metadata_age_cols:
            dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500, hist_tol=0, kde_tol=0), drilldown=True,
                               group="metadata")
        elif col in metadata_float_cols:
            dwc.add_drift_stat(col, KSDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, KdeHistPlotCalculator(npoints=500), drilldown=True, group="metadata")
        elif col in metadata_cat_cols:
            dwc.add_drift_stat(col, ChiSqDriftCalculator(), drilldown=False, group="metadata")
            dwc.add_drift_stat(col, HistIntersectionCalculator(), drilldown=True, group="metadata")

    score_cols = match_keys(list(dataframe), score_cols)
    vae_cols = match_keys(list(dataframe), vae_cols)
    metadata_cols = metadata_age_cols + metadata_cat_cols + metadata_float_cols

    dwc = TabularDriftCalculator()
    for col in dataframe.columns:
        if col in score_cols:
            add_score_metrics(dwc, col)
        elif col in vae_cols:
            add_vae_metrics(dwc, col)
        elif col in metadata_cols:
            add_metadata_metrics(dwc, col)

    return dwc

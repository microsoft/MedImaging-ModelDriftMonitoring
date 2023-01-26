from multiprocessing import Pool
from pathlib import Path


import pandas as pd
from pydicom import dcmread
from pydicom.datadict import tag_for_keyword



kws = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "Modality",
    "PatientID",
    "AccessionNumber",
    "InstanceNumber",
    "SeriesNumber",
    "SeriesDescription",
    "StudyDate",
    "PixelSpacing"
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "PresentationIntentType",
    "BodyPartExamined",
    "KVP",
    "PhotometricInterpretation",
    "PixelRepresentation",
    "PixelAspectRatio",
    "SpatialResolution",
    "BitsAllocated",
    "BitsStored",
    "Rows",
    "Columns",
    "XRayTubeCurrent",
    "Exposure",
    "RelativeXRayExposure",
    "ExposureInuAs",
    "ImagerPixelSpacing",
    "ViewPosition",
    "ImageLaterality",
    "PatientOrientation",
    "WindowCenter",
    "WindowWidth",
]


def process_file(p: Path):
    print(p)
    dcm = dcmread(p, stop_before_pixels=True)
    d = {}
    for kw in kws:
        d[kw] = getattr(dcm, kw, '')

    d["TransferSyntaxUID"] = dcm.file_meta.TransferSyntaxUID
    return d


def make_dicom_inventory():
    for kw in kws:
        assert tag_for_keyword(kw) is not None

    top_dir = Path("/autofs/cluster/qtim/datasets/xray_drift/dicom/studies/")
    files = list(top_dir.glob("**/series/**/instances/*"))

    with Pool(64) as pool:
        results = pool.map(process_file, files)

    results_df = pd.DataFrame(results)
    results_df.to_csv("dicom_inventory.csv")


if __name__ == "__main__":
    make_dicom_inventory()

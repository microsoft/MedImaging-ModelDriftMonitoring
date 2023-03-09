# This file contains paths used for the mgb experiments
from pathlib import Path
import sys


# Standard location for share mounting
# This will vary based on the platform
if sys.platform == "darwin":
    # 'darwin' just means MacOS
    mount_point = Path("/Volumes/")
else:
    # Linux servers and containers
    mount_point = Path("/autofs/cluster/")

# Directories containing data files
data_dir = mount_point / "qtim/datasets/private/xray_drift"
csv_dir = data_dir / "csv"
study_list_csv = csv_dir / "study_list.csv"
dicom_inventory_csv = csv_dir / "dicom_inventory.csv"
labels_csv = csv_dir / "labels.csv"
raw_labels_csv = csv_dir / "raw_labels.csv"
crosswalk_csv = csv_dir / "IRB2022P002646_Crosswalk.csv"
dicom_dir = data_dir / "dicom"
reports_dir = data_dir / "reports"
reports_csv = reports_dir / "combined_reports.csv"
impressions_csv = reports_dir / "impressions.csv"

# Directories containing project files
project_dir = mount_point / "qtim/projects/xray_drift"
drift_dir = project_dir / "drift_analyses"
inference_dir = project_dir / "inferences"
model_dir = project_dir / "models"
analysis_dir = project_dir / "analysis"
report_checking_dir = project_dir / "report_checking"

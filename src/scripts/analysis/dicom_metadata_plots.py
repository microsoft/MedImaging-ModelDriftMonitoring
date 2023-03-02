from ast import literal_eval
from plotnine import (
    ggplot,
    geom_histogram,
    aes,
    position_fill,
    ggtitle,
    theme,
    facet_grid,
    geom_smooth,
)
import pandas as pd
from model_drift import mgb_locations
from pycrumbs import tracked


from pathlib import Path
OUTPUT_DIRECTORY = Path("tmp")
# OUTPUT_DIRECTORY = mgb_locations.analysis_dir / "dicom_metadata"


@tracked(literal_directory=OUTPUT_DIRECTORY)
def dicom_metadata_plots():

    # Read in DICOM inventory
    df = pd.read_csv(
        mgb_locations.dicom_inventory_csv,
        dtype={"AccessionNumber": str, "PatientID": str},
    )

    # Drop incorrect date column
    df.drop(columns="StudyDate", inplace=True)

    # Reports
    reports_df = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    reports_df = reports_df[
        [
            "Accession Number",
            "Point of Care",
            "Patient Sex",
            "Patient Age",
            "Is Stat",
            "Exam Code",
        ]
    ].copy()
    crosswalk_df = pd.read_csv(mgb_locations.crosswalk_csv, dtype=str)
    reports_df = reports_df.merge(
        crosswalk_df,
        left_on="Accession Number",
        right_on="ORIG_AccNumber",
        validate="one_to_one",
    )

    # Read in labels file for the correct date, drop other columns
    labels_df = pd.read_csv(
        mgb_locations.labels_csv
    )[["StudyInstanceUID", "StudyDate"]].copy()
    labels_df = labels_df[labels_df.StudyInstanceUID.notnull()].copy()
    labels_df["StudyDate"] = pd.to_datetime(labels_df.StudyDate)

    assert labels_df.StudyInstanceUID.nunique() == len(labels_df)

    # Join the dataframes to get the correct date
    df = df.merge(
        labels_df,
        on="StudyInstanceUID",
        how="inner",
        validate="many_to_one",
    )

    df["AccessionNumber"] = df.AccessionNumber.apply(lambda x: x.lstrip("0"))

    df = df.merge(
        reports_df,
        left_on="AccessionNumber",
        right_on="ANON_AccNumber",
        validate="many_to_one",
    )

    # Columns of interest
    cols = {
        "ViewPosition": "CAT",
        "Manufacturer": "CAT",
        "ManufacturerModelName": "CAT",
        "PhotometricInterpretation": "CAT",
        "BitsStored": "CAT",
        "Rows": "FLOAT",
        "Columns": "FLOAT",
        "XRayTubeCurrent": "FLOAT",
        "Exposure": "FLOAT",
        "ExposureInuAs": "FLOAT",
        "KVP": "FLOAT",
        "Modality": "CAT",
        "PixelRepresentation": "CAT",
        "PixelAspectRatio": "CAT",
        # "SpatialResolution": "CAT",  # this is empty
        "WindowCenter": "FLOAT",
        "WindowWidth": "FLOAT",
        "RelativeXRayExposure": "FLOAT",
        "Point of Care": "CAT",
        "Exam Code": "CAT",
        "Is Stat": "CAT",
        "Patient Sex": "CAT",
        "Patient Age": "FLOAT",
    }

    def make_float(x):
        if isinstance(x, (float, int)):
            return float(x)
        try:
            v = literal_eval(x)
        except Exception:
            print("bad", x, type(x))
            raise
        if isinstance(v, list):
            v = float(v[0])
        else:
            return float(x)

    # Need to fix some columns that may contain lists
    df["WindowWidth"] = df.WindowWidth.apply(make_float)
    df["WindowCenter"] = df.WindowCenter.apply(make_float)

    # For categorical variables stored as ints, convert to strings so that
    # histogram will work correctly
    df["BitsStored"] = df.BitsStored.astype(str)
    df["PixelRepresentation"] = df.PixelRepresentation.astype(str)

    # Iterate over columns of interest
    for col, col_type in cols.items():
        if col_type == "CAT":
            col_fname = col.replace(" ", "_")
            # Single plot
            p = (
                ggplot(df, aes(x="StudyDate", fill=col)) +
                geom_histogram(position=position_fill) +
                ggtitle(col) +
                theme(figure_size=(10, 6))
            )
            p.save(OUTPUT_DIRECTORY / f"{col_fname}_histogram.png")

            # Per scanner model
            if col not in ("Manufacturer", "ManufacturerModelName"):
                p = (
                    ggplot(df, aes(x="StudyDate", fill=col)) +
                    geom_histogram(position=position_fill) +
                    ggtitle(col) +
                    theme(figure_size=(10, 25)) +
                    facet_grid("ManufacturerModelName ~ .")
                )
                p.save(OUTPUT_DIRECTORY / f"{col_fname}_histogram_by_model.png")

        if col_type == "FLOAT":
            p = (
                ggplot(
                    df,
                    aes(x="StudyDate", y=col),
                ) +
                geom_smooth() +
                ggtitle(col) +
                theme(figure_size=(10, 6))
                # facet_grid('ManufacturerModelName ~ .')
            )
            p.save(OUTPUT_DIRECTORY / f"{col_fname}_smooth.png")

            # Per scanner model
            try:
                p = (
                    ggplot(
                        df,
                        aes(x="StudyDate", y=col, color="ManufacturerModelName"),
                    ) +
                    geom_smooth() +
                    ggtitle(col) +
                    theme(figure_size=(10, 6))
                )
                p.save(OUTPUT_DIRECTORY / f"{col_fname}_smooth_by_model.png")
            except Exception:
                print("Error in", col)
                pass


if __name__ == "__main__":
    dicom_metadata_plots()

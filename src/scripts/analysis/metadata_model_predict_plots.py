from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from model_drift import mgb_locations
from pycrumbs import tracked
import pathlib
import json
import matplotlib.pyplot as plt

OUTPUT_DIRECTORY = mgb_locations.analysis_dir / "metadata_model_predict"


# Process DICOM metadata (code from Chris Bridge)
@tracked(literal_directory=OUTPUT_DIRECTORY)
def process_dicom_metadata():
    # Read in DICOM inventory
    df = pd.read_csv(mgb_locations.dicom_inventory_csv, index_col=0)

    # Drop incorrect date column
    df.drop(columns="StudyDate", inplace=True)

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

    return df


# Process model prediction results
def process_prediction_results():
    # Read in prediction file
    text = pathlib.Path((mgb_locations.inference_dir / "mgb_with_chexpert_model_rangenorm" / "preds.jsonl")).read_text()
    pred_list = [json.loads(s) for s in text.splitlines()]  # this is a list of dictionaries

    # Convert the list into a dataframe
    df = pd.DataFrame(pred_list)

    # Process the index column
    df["PatientID"] = df.loc[:, "index"].apply(lambda x: x.split("_")[0])
    df["AccessionNumber"] = df.loc[:, "index"].apply(lambda x: x.split("_")[1])
    df["SOPInstanceUID"] = df.loc[:, "index"].apply(lambda x: x.split("_")[2])

    # Process the label column - check
    labels = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Lung Lesion',
        'No Finding',
        'Lung Opacity',
        'Pleural Other',  # merge ptx here?
        'Pleural Effusion',
        'Pneumonia',
    ]

    for idx, label in enumerate(labels):
        label_pred = label + '_pred'
        label_true = label + '_true'
        df[label_pred] = df.loc[:, "activation"].apply(lambda x: x[idx])
        df[label_true] = df.loc[:, "label"].apply(lambda x: x[idx])

    # Convert Patient ID and Accession Number to int
    df['PatientID'] = df['PatientID'].astype(int)
    df['AccessionNumber'] = df['AccessionNumber'].astype(int)

    # Drop redundant columns (i.e., index, and label)
    df.drop(columns=["index", "score", "activation", "label"], inplace=True)

    return df


def process_labels_df():
    df = pd.read_csv(mgb_locations.labels_csv, index_col=0)

    # Drop observations with PatientID/AccessionNumber/StudyInstanceUID equal to NaN
    df = df.loc[df.PatientID.notna(), :]

    # Transform PatientID and AccessionNumber to int type
    df.PatientID = df.PatientID.apply(round)
    df.AccessionNumber = df.AccessionNumber.apply(round)

    return df


# Correlation between prediction accuracy and metadata
def correlate_performance_score_and_metadata(metadata, score_name):
    # Create a label list
    labels = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Lung Lesion',
        'No Finding',
        'Lung Opacity',
        'Pleural Other',  # merge ptx here?
        'Pleural Effusion',
        'Pneumonia',
    ]

    # Find unique values in metadata
    indices = list(merged[metadata].unique())

    # Create output dataframe
    df = np.zeros((len(indices), len(labels)))
    df = pd.DataFrame(df, columns=labels, index=indices)

    # Compute f1 score
    for idx in indices:
        subset_merged = merged[merged[metadata] == idx]
        for label in labels:
            true_values = subset_merged[label + '_true']
            pred_values = (subset_merged[label + '_pred'] > 0.5) * 1
            if score_name == "f1":
                score = f1_score(true_values, pred_values)
            elif score_name == "accuracy":
                score = accuracy_score(true_values, pred_values)
            elif score_name == "roc_auc":
                score = roc_auc_score(true_values, subset_merged[
                    label + '_pred'])  # should I instead use the predicted probabilities?
            # Store the computed f1 score
            df.loc[idx, label] = score

    return df


# Prevalence of predicted labels within each sub-categories
def label_prevalence(metadata):
    # metadata refers to the columns name

    # List of predicted labels
    predicted_labels = ['Atelectasis_pred',
                        'Cardiomegaly_pred',
                        'Consolidation_pred',
                        'Edema_pred',
                        'Lung Lesion_pred',
                        'No Finding_pred',
                        'Lung Opacity_pred',
                        'Pleural Other_pred',  # merge ptx here?
                        'Pleural Effusion_pred',
                        'Pneumonia_pred']

    # Transform all predicted labels from probabilities to 0s or 1s
    transformed_pred_df = (merged.loc[:, predicted_labels] > 0.5) * 1

    # Create the data frame for grouping predicted labels
    metadata_df = merged.loc[:, [metadata]]
    df = pd.concat([metadata_df, transformed_pred_df], axis=1)

    # Group predicted labels by the selected metadata
    grouped_labels = df.groupby(by=metadata).sum()

    # Sum grouped labels by rows
    grouped_labels_sum = grouped_labels.sum(axis=1)

    # Compute label prevalence across sub-groups of selected meta data
    label_prevalence = grouped_labels.div(grouped_labels_sum, axis='rows')

    return label_prevalence


# Define Covid19 Plotting
def covid_plotting(type, score="auc_roc", label="Pneumonia"):
    # Read in data
    # covid_cases_by_date = pd.read_excel("mass_gov_covid_data.xlsx",
    #                                     sheet_name="CasesByDate (Test Date)")
    # covid_hospitalization_by_date = pd.read_excel("mass_gov_covid_data.xlsx",
    #                                               sheet_name="Hospitalization from Hospitals")

    # Subset data frame
    covid_cases_by_date = covid_cases_by_date_df.loc[(covid_cases_by_date_df.Date >= "2020-01-29 00:00:00") &
                                                  (covid_cases_by_date_df.Date <= "2021-07-01 00:00:00"), :]
    covid_hospitalization_by_date = covid_hospitalization_by_date_df.loc[
                                    (covid_hospitalization_by_date_df.Date >= "2020-01-29 00:00:00") &
                                    (covid_hospitalization_by_date_df.Date <= "2021-07-01 00:00:00"), :]

    # Create covid plot
    figure, ax = plt.subplots(2, 1, figsize=(18, 12))
    ax[0].plot("Date", "7-day confirmed case average", label="Confirmed cases", data=covid_cases_by_date)
    ax[0].plot("Date", "7 day average of COVID hospitalizations", label="Number of hospitalizations",
               data=covid_hospitalization_by_date)
    ax[0].set_xlabel("Date")
    ax[0].set_title("Covid-19 Trends in Massachusetts")

    print(ax[0].get_xticklabels())
    ax[0].legend()

    # Create x-ray plot
    if type == "number of scans":
        # Create number of scans per day plot
        number_scans_by_date = merged.loc[:, ["StudyDate", "SOPInstanceUID"]]. \
            groupby(by="StudyDate", as_index=False).count()
        number_scans_by_date = number_scans_by_date[number_scans_by_date.StudyDate >= '2020-01-29 00:00:00']
        number_scans_by_date["# Scans 7-Day Moving Average"] = number_scans_by_date.rolling(7).mean()

        ax[1].plot("StudyDate", "# Scans 7-Day Moving Average",
                   label="7-Day Moving Average",
                   data=number_scans_by_date)
        ax[1].set_title("Plot on Number of Scans")
        ax[1].legend()

    elif type == "label prevalence":
        # Create label prevalence data frame
        label_prevalence_percentage = merged.loc[:, ["StudyDate", (label + "_true")]].groupby(by="StudyDate", as_index=False).mean()
        label_prevalence_percentage = label_prevalence_percentage.rename(columns={(label + "_true"): "Percentage"})
        label_prevalence_percentage = label_prevalence_percentage[label_prevalence_percentage.StudyDate >= '2020-01-29 00:00:00']
        label_prevalence_value = merged.loc[:, ["StudyDate", (label + "_true")]].groupby(by="StudyDate", as_index=False).sum()
        label_prevalence_value = label_prevalence_value.rename(columns={(label + "_true"): "Value"})
        label_prevalence_value = label_prevalence_value[label_prevalence_value.StudyDate >= '2020-01-29 00:00:00']
        label_prevalence_df = pd.merge(label_prevalence_percentage, label_prevalence_value, on="StudyDate")
        label_prevalence_df["14-Day Moving Average Percentage"] = label_prevalence_df["Percentage"].rolling(14).mean()
        label_prevalence_df["14-Day Moving Average Value"] = label_prevalence_df["Value"].rolling(14).mean()

        # Plot label prevalence ratio and value
        p1, = ax[1].plot("StudyDate", "14-Day Moving Average Percentage", color='red', data=label_prevalence_df)
        ax[1].set_ylabel("Percentage of Scans with " + label)
        ax[1].set_title(label + " Label Prevalence Trends")

        ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
        p2, = ax2.plot("StudyDate", "14-Day Moving Average Value", color='orange', data=label_prevalence_df)
        ax2.set_ylabel("Number of Scans with " + label)
        plt.legend([p1, p2], ["14-Day Moving Average Percentage", "14-Day Moving Average Value"])

    elif type == "performance by label":
        # Note: performance|label name|auroc|obs is the name of AUC-ROC score for each label
        # Note: performance|label name|f1-score|obs is the name of F1 score for each label

        # Create model performance by label data frame
        auroc_score = "performance|" + label + "|auroc|obs"
        f1_score = "performance|" + label + "|f1-score|obs"
        label_performance_df = output_df.loc[output_df.StudyDate >= "2020-01-29", ["StudyDate", auroc_score, f1_score]]

        # Plot AUC-ROC and F1-Score of model performance by label
        p1, = ax[1].plot("StudyDate", auroc_score, color='red', data=label_performance_df)
        ax[1].set_ylabel("AUC-ROC Score (" + label + ")")
        ax[1].set_title(label + " Label Performance Trends")
        ticks = [32, 93, 154, 216, 277, 338, 397, 458, 519]
        tick_labels = ['2020-03', '2020-05', '2020-07', '2020-09', '2020-11', '2021-01', '2021-03', '2021-05', '2021-07']
        ax[1].set_xticks(ticks, labels=tick_labels)
        # print("ax1", ax[1].get_xticklabels())

        ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
        p2, = ax2.plot("StudyDate", f1_score, color='orange', data=label_performance_df)
        ax2.set_ylabel("F1-Score (" + label + ")")
        # print("ax2", ax2.get_xticklabels())
        plt.legend([p1, p2], ["AUC-ROC Score", "F1 Score"])

    elif type == "micro average":
        # Note: performance.220 is auc-roc mean values
        # Note: performance.224 is f1-score mean values

        if score == "auc_roc":
            performance = "performance|micro avg|auroc|mean"
            label = "AUC ROC Mean Values"
        elif score == "f1_score":
            performance = "performance|micro avg|f1-score|mean"
            label = "F1 Score Mean Values"

        # Create micro average data frame
        date_df = pd.DataFrame(merged.StudyDate.unique()).\
            rename(columns={0: "Date"}).\
            sort_values(by="Date").\
            reset_index(drop=True)
        micro_avg = output_df[[performance]].iloc[3:, :].reset_index(drop=True)
        micro_avg_df = pd.concat([date_df, micro_avg], axis=1)

        micro_avg_df = micro_avg_df[micro_avg_df.Date >= '2020-01-29 00:00:00']
        micro_avg_df[performance] = micro_avg_df[performance].astype(float)

        # Plot micro average plot
        ax[1].plot("Date", performance, label=label, data=micro_avg_df)
        ax[1].set_title("Micro-average Plot")
        ax[1].legend()

    elif type == "macro average":
        # Note: performance.200 is auc-roc mean values
        # Note: performance.204 is f1-score mean values

        if score == "auc_roc":
            performance = "performance|macro avg|auroc|mean"
            label = "AUC ROC Mean Values"
        elif score == "f1_score":
            performance = "performance|macro avg|f1-score|mean"
            label = "F1 Score Mean Values"

        # Create macro average data frame
        date_df = pd.DataFrame(merged.StudyDate.unique()). \
            rename(columns={0: "Date"}). \
            sort_values(by="Date"). \
            reset_index(drop=True)
        macro_avg = output_df[[performance]].iloc[3:, :].reset_index(drop=True)
        macro_avg_df = pd.concat([date_df, macro_avg], axis=1)

        macro_avg_df = macro_avg_df[macro_avg_df.Date >= '2020-01-29 00:00:00']
        macro_avg_df[performance] = macro_avg_df[performance].astype(float)

        # Plot macro average plot
        ax[1].plot("Date", performance, label=label, data=macro_avg_df)
        ax[1].set_title("Macro-average Plot")
        ax[1].legend()

    figure.tight_layout()
    plt.show()


# Define MMC plotting
def mmc_plotting(weight="unweighted"):
    # Covid19 data
    covid_cases_by_date = pd.read_excel("mass_gov_covid_data.xlsx", sheet_name="CasesByDate (Test Date)")
    covid_hospitalization_by_date = pd.read_excel("mass_gov_covid_data.xlsx",
                                                  sheet_name="Hospitalization from Hospitals")

    # Subset data frame
    covid_cases_by_date = covid_cases_by_date.loc[(covid_cases_by_date.Date >= "2020-01-29 00:00:00") &
                                                  (covid_cases_by_date.Date <= "2021-07-01 00:00:00"), :]
    covid_hospitalization_by_date = covid_hospitalization_by_date.loc[
                                    (covid_hospitalization_by_date.Date >= "2020-01-29 00:00:00") &
                                    (covid_hospitalization_by_date.Date <= "2021-07-01 00:00:00"), :]

    # Create data frame for mmc plot
    mmc_col_list = [col for col in output_df
                    if (not col.startswith('performance')) &
                    col.endswith("|critical_diff|mean")]
    mmc_df = output_df.loc[:, mmc_col_list]

    # Normalize the columns
    normalized_mmc_df = mmc_df.apply(lambda col: (col - col.mean()) / col.std(), axis=0)

    # Compute MMC
    num_metric = normalized_mmc_df.shape[1]
    if weight == "unweighted":
        normalized_mmc_df["unweighted mmc"] = normalized_mmc_df.sum(axis=1) / num_metric
    else:
        pass

    # Select the data by date
    normalized_mmc_df["StudyDate"] = output_df["StudyDate"]
    normalized_mmc_df = normalized_mmc_df.loc[(normalized_mmc_df.StudyDate >= "2020-01-29") &
                                              (normalized_mmc_df.StudyDate <= "2021-07-01"), :]

    # Plot Covid information and MMC
    figure, ax = plt.subplots(2, 1, figsize=(18, 12))
    ax[0].plot("Date", "7-day confirmed case average", label="Confirmed cases", data=covid_cases_by_date)
    ax[0].plot("Date", "7 day average of COVID hospitalizations", label="Number of hospitalizations",
               data=covid_hospitalization_by_date)
    ax[0].set_xlabel("Date")
    ax[0].set_title("Covid-19 Trends in Massachusetts")
    ax[0].legend()

    ax[1].plot("StudyDate", "unweighted mmc", label=weight, data=normalized_mmc_df)
    ax[1].set_title("Unweighted MMC Plot")
    ticks = [32, 93, 154, 216, 277, 338, 397, 458, 519]
    tick_labels = ['2020-03', '2020-05', '2020-07', '2020-09', '2020-11', '2021-01', '2021-03', '2021-05', '2021-07']
    ax[1].set_xticks(ticks, label=tick_labels)
    ax[1].legend()


if __name__ == "__main__":
    # Load all data frames
    dcm_df = process_dicom_metadata()
    results_df = process_prediction_results()
    labels_df = process_labels_df()
    output_df = pd.read_csv(mgb_locations.drift_dir / "initial_full_tags_take2" / "output.csv", header=[0,1,2,3])
    output_df.columns = output_df.columns.map('|'.join).str.strip('|')
    output_df = output_df.rename(columns={'Unnamed: 0_level_0|Unnamed: 0_level_1|Unnamed: 0_level_2|Unnamed: 0_level_3':'StudyDate'})

    covid_data_file_path = "../data/mass_gov_covid_data.xlsx"  # Use your path to covid data here.
    covid_cases_by_date_df = pd.read_excel(covid_data_file_path, sheet_name="CasesByDate (Test Date)")
    covid_hospitalization_by_date_df = pd.read_excel(covid_data_file_path, sheet_name="Hospitalization from Hospitals")

    # Merge results and dicom data frames together
    merged = results_df.merge(dcm_df, on=("PatientID", "AccessionNumber", "SOPInstanceUID"))

    # Create plots
    covid_plotting(type="number of scans")
    covid_plotting(type="micro average")
    covid_plotting(type="micro average", score="f1_score")
    covid_plotting(type="macro average")
    covid_plotting(type="macro average", score="f1_score")

    labels = ['Atelectasis',
              'Cardiomegaly',
              'Consolidation',
              'Edema',
              'Lung Lesion',
              'No Finding',
              'Lung Opacity',
              'Pleural Other',
              'Pleural Effusion',
              'Pneumonia']
    for label in labels:
        covid_plotting(type="label prevalence", label=label)
        covid_plotting(type="performance by label", label=label)

    covid_plotting(type="performance by label", label="Pneumonia")

    mmc_plotting()

############################################### Some Sanity Checks #####################################################
    # Compute predicted label prevalence
    manufacturer_label_prevalence = label_prevalence("Manufacturer")
    manufacturer_model_name_label_prevalence = label_prevalence("ManufacturerModelName")
    photometric_label_prevalence = label_prevalence("PhotometricInterpretation")

    # Compute correlation between F1-score and metadata
    manufacturer_f1_scores = correlate_performance_score_and_metadata("Manufacturer", "f1")
    manufacturer_model_name_f1_scores = correlate_performance_score_and_metadata("ManufacturerModelName", "f1")
    photometric_f1_scores = correlate_performance_score_and_metadata("PhotometricInterpretation", "f1")

    # Compute correlation between accuracy scores and metadata
    manufacturer_accuracy_scores = correlate_performance_score_and_metadata("Manufacturer", "accuracy")
    manufacturer_model_name_accuracy_scores = correlate_performance_score_and_metadata("ManufacturerModelName",
                                                                                       "accuracy")
    photometric_accuracy_scores = correlate_performance_score_and_metadata("PhotometricInterpretation", "accuracy")

    # Compute correlation between auc-roc scores and metadata
    # manufacturer_roc_auc_scores = correlate_performance_score_and_metadata("Manufacturer", "roc_auc")
    # manufacturer_model_name_roc_auc_scores = correlate_performance_score_and_metadata("ManufacturerModelName","roc_auc")
    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    photometric_roc_auc_scores = correlate_performance_score_and_metadata("PhotometricInterpretation", "roc_auc")

    # Check on No Finding label
    ## When all the other labels are 0, what is the proportion of no finding that is 1?
    true_labels = ['Atelectasis_true',
                   'Cardiomegaly_true',
                   'Consolidation_true',
                   'Edema_true',
                   'Lung Lesion_true',
                   'No Finding_true',
                   'Lung Opacity_true',
                   'Pleural Other_true',
                   'Pleural Effusion_true',
                   'Pneumonia_true']
    check_no_finding_df1 = merged.loc[(merged.Atelectasis_true == 0) & (merged.Cardiomegaly_true == 0) &
                                      (merged.Consolidation_true == 0) & (merged.Edema_true == 0) &
                                      (merged["Lung Lesion_true"] == 0) &
                                      (merged["Lung Opacity_true"] == 0) & (merged["Pleural Other_true"] == 0) &
                                      (merged["Pleural Effusion_true"] == 0) & (
                                                  merged.Pneumonia_true == 0), true_labels]
    check_no_finding_df1["No Finding_true"].sum() / check_no_finding_df1.shape[0]

    # Look at some impressions with all labels equal to 0 (including the No Finding label)
    all_zero_labels = merged.loc[(merged.Atelectasis_true == 0) & (merged.Cardiomegaly_true == 0) &
                                 (merged.Consolidation_true == 0) & (merged.Edema_true == 0) &
                                 (merged["Lung Lesion_true"] == 0) & (merged["No Finding_true"] == 0) &
                                 (merged["Lung Opacity_true"] == 0) & (merged["Pleural Other_true"] == 0) &
                                 (merged["Pleural Effusion_true"] == 0) & (merged.Pneumonia_true == 0),
    ['PatientID', 'AccessionNumber', 'StudyInstanceUID']]
    ## impressions from the labels_df
    all_zeros_labels_reports = pd.merge(all_zero_labels, labels_df,
                                        on=['PatientID', 'AccessionNumber', 'StudyInstanceUID'])
    all_zeros_labels_reports = all_zeros_labels_reports.Reports

    # When the No Finding label is 1, what is the proportion of observations with some other labels equal to 1
    check_no_finding_df2 = merged.loc[(merged["No Finding_true"] == 1), true_labels]
    check_no_finding_df2_subset = check_no_finding_df2.loc[(check_no_finding_df2.Atelectasis_true == 1) |
                                                           (check_no_finding_df2.Cardiomegaly_true == 1) |
                                                           (check_no_finding_df2.Consolidation_true == 1) |
                                                           (check_no_finding_df2.Edema_true == 1) |
                                                           (check_no_finding_df2["Lung Lesion_true"] == 1) |
                                                           (check_no_finding_df2["Lung Opacity_true"] == 1) |
                                                           (check_no_finding_df2["Pleural Other_true"] == 1) |
                                                           (check_no_finding_df2["Pleural Effusion_true"] == 1) |
                                                           (check_no_finding_df2.Pneumonia_true == 1), :]

    # Check on Manufacturer & Manufacturer Model Name with only one class
    ## Number of unique values under each subgroup
    print(merged.loc[:, ["Manufacturer", "PatientID"]].groupby(by="Manufacturer").count().sort_values(by="PatientID"))
    print(merged.loc[:, ["ManufacturerModelName", "PatientID"]].groupby(by="ManufacturerModelName").count().sort_values(
        by="PatientID"))
    print(merged.loc[:, ["PhotometricInterpretation", "PatientID"]].groupby(
        by="PhotometricInterpretation").count().sort_values(by="PatientID"))

    ## Number of unique values by subgroup and labels
    manufacturer_check_labels_dict = {}
    manufacturer_model_name_check_labels_dict = {}
    photometric_check_labels_dict = {}

    for label in true_labels:
        manufacturer_check_labels_dict[label] = merged.loc[:, ["Manufacturer", label, "PatientID"]]. \
            groupby(by=["Manufacturer", label]).count()
        # Note: SIEMENS is the only manufacturer with one class (only 0s or 1s) under some labels
        manufacturer_model_name_check_labels_dict[label] = merged.loc[:, ["ManufacturerModelName", label, "PatientID"]]. \
            groupby(by=["ManufacturerModelName", label]).count()
        # Note: Fluorospot Compact FD is the only manufacturer model name with one class (only 0s or 1s) under some labels
        photometric_check_labels_dict[label] = merged.loc[:, ["PhotometricInterpretation", label, "PatientID"]]. \
            groupby(by=["PhotometricInterpretation", label]).count()

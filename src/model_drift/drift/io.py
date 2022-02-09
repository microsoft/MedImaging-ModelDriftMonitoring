import pandas as pd

from model_drift.helpers import column_xs


def load_weights(fn):
    return pd.read_csv(fn, index_col=[0,1,2]).iloc[:,0]

def load_stats(fn):
    return pd.read_csv(fn, index_col=[0], header=[0,1,2])


def load_metrics_file(fname, which="mean"):

    combined_df = pd.read_csv(str(fname), index_col=0, header=[0, 1, 2, 3])
    combined_df.index = pd.to_datetime(combined_df.index)
    flip = column_xs(combined_df, include=["pval"])
    combined_df[flip] = 1 - combined_df[flip]

    error_df = combined_df.swaplevel(0, -1, axis=1)[["std"]].swaplevel(0, -1, axis=1).droplevel(-1, axis=1).copy()
    combined_df = combined_df.swaplevel(0, -1, axis=1)[[which]].swaplevel(0, -1, axis=1).droplevel(-1, axis=1).copy()

    return error_df, combined_df

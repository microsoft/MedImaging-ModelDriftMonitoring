import pandas as pd
from distutils import dir_util
from . import settings
from .utils.padchest import fix_strlst


def read_padchest(csv_file=None) -> pd.DataFrame:
    csv_file = csv_file or settings.PADCHEST_FILENAME
    df = pd.read_csv(csv_file, low_memory=False, index_col=0)
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    df["Labels"] = fix_strlst(df["Labels"])
    return df


def prepare_padchest(df) -> pd.DataFrame:
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    return df


def rolling_dt_apply_with_stride(
    dataframe,
    function,
    window="30D",
    stride="D",
    unique_only=False,
    center=False,
    min_periods=None,
) -> pd.DataFrame:
    if unique_only:
        tmp_index = dataframe.index.unique()
    else:
        tmp_index = pd.date_range(dataframe.index.min(), dataframe.index.max(), freq=stride)

    try:
        delta = pd.tseries.frequencies.to_offset(window)
        fdelta = (delta / 2) if center else pd.tseries.frequencies.to_offset(window) * 0
        bdelta = delta - fdelta
    except TypeError as e:
        raise ValueError("Centering does not work with all windows and strides") from e

    def _apply(i):
        window = dataframe[i - bdelta : i + fdelta]
        if min_periods is not None and len(window) < min_periods:
            return None
        return window.agg(function)

    return pd.concat({i: _apply(i) for i in tmp_index}, axis=0).unstack()


def copytree(src, dst):
    dir_util.copy_tree(str(src), str(dst))


def fix_multiindex(out):
    def __make_tuple(k, size):
        outk = [""] * size
        if isinstance(k, tuple):
            outk[: len(k)] = list(k)
        else:
            outk[0] = k
        return tuple(outk)

    out = nested2tuplekeys(out)
    tuple_keys = [len(k) for k in out.keys() if isinstance(k, tuple)] + [0]
    max_tuple_len = max(tuple_keys)
    if not max_tuple_len:
        return out
    return {__make_tuple(k, max_tuple_len): v for k, v in out.items()}


def nested2series(out, name=None):
    return pd.Series(fix_multiindex(out), name=name)


def nested2tuplekeys(out):
    def __fixkey(k, name):
        if isinstance(name, tuple):
            name = list(name)

        if not isinstance(name, list):
            name = [name]

        if isinstance(k, tuple):
            k = list(k)

        if not isinstance(k, list):
            k = [k]

        return tuple(name + k)

    if not isinstance(out, dict):
        return out
    out2 = {}
    for stat_name, nested in out.items():
        nested = nested2tuplekeys(nested)
        if not isinstance(nested, dict):
            out2[stat_name] = nested
            continue
        out2.update({__fixkey(k, stat_name): v for k, v in nested.items()})

    return out2

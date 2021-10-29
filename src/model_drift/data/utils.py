import itertools

import ast
import pandas as pd
import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

tqdm_func = tqdm.tqdm


def fix_strlst(series, clean=True):
    def convert_literal_list(val):
        val = str(val)
        if not len(val):
            return []
        val = ast.literal_eval(val)
        if clean:
            val = [x.strip() for x in val]
        return val

    return series.fillna("[]").apply(convert_literal_list)


def khot_labels(labels):
    khot = MultiLabelBinarizer()
    binary = khot.fit_transform(labels)
    return pd.DataFrame(binary, columns=khot.classes_, index=labels.index)


def binarize_label(series):
    return khot_labels(fix_strlst(series))


def check_label_map(lblmap):
    # quick check
    x = list(itertools.chain(*lblmap.values()))
    y = set(x)
    print(f"Mapping {len(y)} labels into {len(lblmap)} new labels")
    if len(x) == len(y):
        print("No Overlap!")
        return
    cross = {}
    print("Label Overlap")
    for k1, v1 in lblmap.items():
        for k2, v2 in lblmap.items():
            if k1 == k2:
                continue
            a = set(v1).intersection(v2)
            cross[(k1, k2)] = a
            if len(a):
                print(f"   {k1}, {k2}", a)


def remap_label_list(lbllst, label_map):
    def _convert_label(lbl):
        return [k for k, l in label_map.items() if lbl in l]

    out = set()
    for lbl in lbllst:
        out = out.union(_convert_label(lbl))
    return list(out)


def remap_labels(labels, label_map=None, verbose=False):

    if label_map is None:
        label_map_func = lambda x: x

    if isinstance(label_map, dict):
        label_map_func = lambda x: remap_label_list(x, dict(label_map))

    assert callable(label_map_func), "label_map must be a function, dictionary or None"

    labels = fix_strlst(labels)

    prev_count = (labels.apply(len) > 0).sum()
    mapped = labels.transform(label_map_func)
    mapped_count = (mapped.apply(len) > 0).sum()
    if verbose:
        print(f"total: {len(labels)}, # with labels: {prev_count} (original), {mapped_count} (remapped).")
    return mapped


def split_on_date(df, splits, col="StudyDate"):
    splits = pd.to_datetime(["2014-01-01", "2013-01-01"]).sort_values()
    rem = df
    for split in splits:
        curr, rem = rem[rem[col] < split], rem[rem[col] >= split]
        yield curr
    yield rem


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


def rolling_window_dt_apply(dataframe, func, window='30D', stride='D', min_periods=1):
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError()

    tmp_index = pd.date_range(
        dataframe.index.min(), dataframe.index.max(), freq=stride
    )

    delta = pd.tseries.frequencies.to_offset(window)
    fdelta = pd.tseries.frequencies.to_offset(window) * 0
    bdelta = delta - fdelta

    def _apply(i):
        x = dataframe.loc[str(i - bdelta):str(i + fdelta)]
        if len(x) < min_periods:
            return None
        preds = func(x)
        preds["count"] = len(x)
        return nested2series(preds)

    out = {}
    for i in tqdm_func(tmp_index):
        out[i] = _apply(i)
    return pd.concat(out, axis=0).unstack(level=0).T

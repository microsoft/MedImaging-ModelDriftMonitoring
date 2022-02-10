import ast
import itertools

import pandas as pd
import tqdm
from joblib import delayed
from sklearn.preprocessing import MultiLabelBinarizer

from model_drift.helpers import ProgressParallel as Parallel


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
    # fix label map so it can be used on itself
    # for k in label_map:
    #     if k not in label_map[k]:
    #         label_map[k].append(k)

    if label_map is None:
        def label_map_func(x): return x

    if isinstance(label_map, dict):
        def label_map_func(x): return remap_label_list(x, dict(label_map))  # noqa

    assert callable(label_map_func), "label_map must be a function, dictionary or None"

    labels = fix_strlst(labels)

    prev_count = (labels.apply(len) > 0).sum()
    mapped = labels.transform(label_map_func)
    mapped_count = (mapped.apply(len) > 0).sum()
    if verbose:
        print(f"total: {len(labels)}, # with labels: {prev_count} (original), {mapped_count} (remapped).")
    return mapped


def split_on_date(df, splits, col="StudyDate"):
    splits = pd.to_datetime(splits).sort_values()
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


def rolling_window_dt_apply(dataframe, func, window='30D', stride='D', min_periods=1, include_count=True, n_jobs=1,
                            verbose=0, backend='loky', refresh_rate=None, **kwargs):
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
        if include_count:
            preds["window_count"] = len(x)
        return nested2series(preds)

    kwargs['desc'] = f"{tmp_index.min().date()} - {tmp_index.max().date()} window: {window}, stride: {stride}"
    if refresh_rate is not None:
        kwargs['miniters'] = max(int(len(tmp_index) * refresh_rate), 1)
    if n_jobs > 1:
        def job(i):
            return (i, _apply(i))

        out = dict(Parallel(n_jobs=n_jobs,
                            verbose=verbose,
                            backend=backend, total=len(tmp_index), tqdm_kwargs=kwargs)(
            delayed(job)(i) for i in tmp_index))

    else:
        out = {}
        with tqdm.tqdm(tmp_index, **kwargs) as pbar:
            for i in pbar:
                pbar.set_postfix_str(str(i.date()))
                out[i] = _apply(i)
    return pd.concat(out, axis=0).unstack(level=0).T

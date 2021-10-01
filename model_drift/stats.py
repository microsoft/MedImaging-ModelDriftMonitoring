import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare
from scipy.stats import norm


def proportions_diff_z_stat_ind(ref, curr):
    n1 = len(ref)
    n2 = len(curr)

    p1 = float(sum(ref)) / n1
    p2 = float(sum(curr)) / n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - norm.cdf(z_stat)


def calc_p_real(train, test):
    if len(test):
        return ks_2samp(train, test)[1]
    return None


def calc_p_categorical(train, test):
    ref_feature_vc = train[np.isfinite(train)].value_counts()
    current_feature_vc = test[np.isfinite(test)].value_counts()

    keys = set(list(train[np.isfinite(train)].unique()) +
               list(test[np.isfinite(test)].unique()))

    ref_feature_dict = dict.fromkeys(keys, 0)
    for key, item in zip(ref_feature_vc.index, ref_feature_vc.values):
        ref_feature_dict[key] = item

    current_feature_dict = dict.fromkeys(keys, 0)
    for key, item in zip(current_feature_vc.index, current_feature_vc.values):
        current_feature_dict[key] = item

    try:
        if len(keys) > 2:
            f_exp = [value[1] for value in sorted(ref_feature_dict.items())]
            f_obs = [value[1]
                     for value in sorted(current_feature_dict.items())]
            # CHI2 to be implemented for cases with different categories
            p_value = chisquare(f_exp, f_obs)[1]
        else:
            ordered_keys = sorted(list(keys))
            p_value = proportions_diff_z_test(proportions_diff_z_stat_ind(train.apply(lambda x: 0 if x == ordered_keys[0] else 1),
                                                                          test.apply(lambda x: 0 if x == ordered_keys[0] else 1)))
    except ZeroDivisionError as e:
        return None

    return p_value


def calculate_alerts(df_roll, test):
    x = test.astype(int).diff()
    x = x[x != 0].index.tolist()
    if ~test.iloc[0]:
        x = x[1:]
    edges = list(zip(x[::2], x[1::2]))
    alerts = []
    for edge in edges:
        drill_start = edge[0]
        drill_end = edge[1]
        df_drill = df_roll[(df_roll.index >= drill_start)
                           & (df_roll.index <= drill_end)]

        high = df_drill.max()
        low = df_drill.min()

        dfalert = pd.DataFrame([(df_drill.index.min(), high, low), (df_drill.index.max(
        ), high, low), ], columns=['StudyDate', 'h', 'l']).set_index("StudyDate")
        alerts.append(dfalert.copy())

    return alerts

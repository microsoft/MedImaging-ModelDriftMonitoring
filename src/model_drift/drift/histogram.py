#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from model_drift.drift import BaseDriftCalculator


def histogram_intersection(hist1: Counter, hist2: Counter) -> float:
    keys = set(hist1.keys()).union(hist2.keys())
    return sum([max(hist1.get(k, 0), hist2.get(k, 0)) - min(hist1.get(k, 0), hist2.get(k, 0)) for k in keys])


class KdeHistPlotCalculator(BaseDriftCalculator):
    name = "kdehistplot"

    def __init__(self, stat="density", bins=12, hist_tol=0.1, kde_tol=0.2, add_overflow_bins=True,
                 normalization="probability", bw_method="scott", npoints=500, use_ref_kde_bounds=True, **kwargs):
        super().__init__(**kwargs)

        self.stat = stat
        self.bins = bins
        self.hist_tol = hist_tol
        self.kde_tol = kde_tol
        self.add_overflow_bins = add_overflow_bins
        self.normalization = normalization
        self.bw_method = bw_method
        self.npoints = npoints
        self.use_ref_kde_bounds = True
        self._ref_kde_x = None

    @property
    def _hist_calc_density(self):
        return self.stat == "density"

    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")

    def _get_bin_edges(self, data):
        if np.ndim(self.bins) == 0:
            return self.get_edges(data, tol=self.hist_tol, npoints=self.bins,
                                  add_overflow=self.add_overflow_bins)
        if np.ndim(self.bins) == 1:
            return np.histogram_bin_edges(self._ref, bins=self.bins)

    def _numeric_histogram(self, data, edges=None):
        edges = self._get_bin_edges(data) if edges is None else edges
        hist, edges = np.histogram(data, density=self._hist_calc_density, bins=edges)

        if self.normalization == "probability":
            hist = hist.astype(float) / hist.sum()

        widths, centers = self._get_widths_and_centers(edges)
        plot_edges = self._get_plot_edges(edges, widths)
        plot_widths, plot_centers = self._get_widths_and_centers(plot_edges)

        return {
            "hist": hist.astype(float),
            "widths": widths,
            "centers": centers,
            "edges": edges,
            "plot_widths": plot_widths,
            "plot_centers": plot_centers,
            "plot_edges": plot_edges,
            "hist_norm": (hist * plot_widths).sum()
        }

    def _histogram(self, sample, edges=None):
        return self._numeric_histogram(sample, edges=edges)

    def _predict(self, sample):
        res = self._histogram(sample, edges=self._ref_res["edges"])
        res.update(self._kde(sample, norm=res["hist_norm"]))
        res["stat"] = self.stat
        res["normalization"] = self.normalization
        return res

    def _get_plot_edges(self, edges, widths):
        median_width = np.median(widths)
        widths_m = widths / median_width
        ov = widths_m / np.mean(widths_m[1:-1])

        plot_edges = edges.copy()

        if ov[0] > 1e10:
            plot_edges[0] = edges[1] - median_width

        if ov[-1] > 1e10:
            plot_edges[-1] = edges[-2] + median_width

        return plot_edges

    @staticmethod
    def get_edges(*data, tol=0.2, npoints=100, add_overflow=False):
        xmin, xmax = zip(*[(d.min(), d.max()) for d in data])
        xmin, xmax = np.min(xmin), np.max(xmax)
        dx = tol * (xmax - xmin)
        xmin -= dx
        xmax += dx
        edges = np.linspace(xmin, xmax, npoints)
        if add_overflow:
            edges = np.append(np.insert(edges, 0, -1e308), 1e308)
        return edges

    @staticmethod
    def _get_widths_and_centers(edges):
        widths = np.diff(edges)
        return widths, edges[:-1] + widths / 2
    
    def get_x(self, sample):
        if self.use_ref_kde_bounds:
            if self._ref_kde_x is None:
                self._ref_kde_x = self.get_edges(self._ref, tol=self.kde_tol, npoints=self.npoints)
            return self._ref_kde_x
        return self.get_edges(sample, self._ref, tol=self.kde_tol, npoints=self.npoints)

    def _kde(self, sample, norm=1.0):

        sample = sample.dropna()
        try:
            sample_kde = gaussian_kde(sample, bw_method=self.bw_method)
        except:
            return {}

        x = self.get_x(sample)
        kde_ref = self._ref_kde(x) * self._ref_res["hist_norm"]
        kde_sample = sample_kde(x) * norm

        idx = sorted(np.argwhere(np.diff(np.sign(kde_ref - kde_sample))).flatten())
        area = np.trapz(np.minimum(kde_ref, kde_sample), x)

        return {
            "kde_intersection": area,
            "kde": kde_sample,
            "kde_x": x,
            "kde_intersection_x": x[idx],
            "kde_intersection_y": kde_sample[idx]
        }

    def prepare(self, ref):
        ref = self.convert(ref)
        self._ref_res = self._histogram(ref)
        self._ref_kde = gaussian_kde(ref.dropna(), bw_method=self.bw_method)
        super().prepare(ref)


class HistIntersectionCalculator(BaseDriftCalculator):
    name = "histogram"

    def convert(self, arg):
        return arg.apply(str)

    def prepare(self, ref):
        ref = self.convert(ref)
        self._ref_counts = Counter(ref)
        refv = np.array(list(self._ref_counts.values()))
        self._ref_norm = dict(zip(ref.keys(), refv / refv.sum()))
        super().prepare(ref)

    def _predict(self, sample):
        sample_counts = Counter(sample)
        keys = sorted(set(sample_counts.keys()).union(self._ref_counts.keys()))
        obs = np.array([sample_counts.get(k, 0) for k in keys])
        obs_norm = obs / obs.sum()

        out = {
            "x": list(map(str, keys)),
            "count": obs,
            "probability": obs_norm,
            "intersection": histogram_intersection(sample_counts, self._ref_counts),
            "normalized_intersection": histogram_intersection(dict(zip(keys, obs_norm)), self._ref_norm)
        }

        return out

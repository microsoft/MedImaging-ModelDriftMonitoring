from collections import Counter, Iterable

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from model_drift.drift import BaseDriftCalculator
def kde_intersection_area(ref, sample, method="scott", tol=0.2, npoints=100, use_prob=False):
    
    ref = ref.dropna()
    sample = sample.dropna()
    
    kde1 = gaussian_kde(ref, bw_method=method)
    kde2 = gaussian_kde(sample, bw_method=method)

    xmin = min(ref.min(), sample.min())
    xmax = max(ref.max(), sample.max())
    dx = tol * (xmax - xmin)
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, npoints)
    
    dt = np.diff(x).mean()
        
    kde1_x = kde1(x)
    kde2_x = kde2(x)
    kde1_nrm = dt
    kde2_nrm = dt
    if use_prob:
        kde1_nrm = 1/np.trapz(kde1_x, x)
        kde2_nrm = 1/np.trapz(kde2_x, x)
    
    print(kde1_nrm, kde2_nrm)
        
    kde1_x *= kde1_nrm
    kde2_x *= kde2_nrm
    
    idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()
    idx = sorted(idx)

    area = np.trapz(np.minimum(kde1_x, kde2_x), x)
    return area, x[idx], kde1_x[idx], x, kde1_x, kde2_x

def histogram_intersection(hist1 : Counter, hist2: Counter) -> float:
    keys = set(hist1.keys()).union(hist2.keys())
    return sum([max(hist1.get(k,0), hist2.get(k,0))-min(hist1.get(k,0), hist2.get(k,0)) for k in keys])



class KdeHistPlotCalculator(BaseDriftCalculator):
    name = "kdehistplot"
    
    def __init__(self, stat="density", bins=12, hist_tol=0.1, kde_tol=0.01, add_overflow_bins=True, normalization="probability", bw_method="scott", npoints=200, **kwargs):
        super().__init__(**kwargs)
        
        self.stat = stat
        self.bins = bins
        self.hist_tol = hist_tol
        self.kde_tol = kde_tol
        self.add_overflow_bins = add_overflow_bins 
        self.normalization = normalization
        self.bw_method = bw_method
        self.npoints = npoints
        
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
            "hist_norm": (hist*plot_widths).sum()
        }

    

    def _histogram(self, sample, edges=None):
        return self._numeric_histogram(sample, edges=edges)
            
    def _predict(self, sample):
        res = self._histogram(sample, edges = self._ref_res["edges"])
        res.update(self._kde(sample, norm=res["hist_norm"]))
        res["stat"] = self.stat
        res["normalization"] = self.normalization
        return res
        
        

    def _get_plot_edges(self, edges, widths):
        median_width = np.median(widths)
        widths_m = widths/median_width
        ov = widths_m/np.mean(widths_m[1:-1])

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
            edges = np.append(np.insert(edges,0,-1e308), 1e308)
        return edges

    @staticmethod
    def _get_widths_and_centers(edges):
        widths = np.diff(edges)
        return widths, edges[:-1] + widths / 2

    def _kde(self, sample, norm=1.0):
        
        sample_kde = gaussian_kde(sample.dropna(), bw_method=self.bw_method)
        x = self.get_edges(sample, self._ref, self._ref, tol=self.kde_tol, npoints=self.npoints)
        
        kde_ref = self._ref_kde(x)*self._ref_res["hist_norm"]
        kde_sample = sample_kde(x)*norm
        
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
        ## calculate ref kde+histogram
        

class HistIntersectionCalculator(BaseDriftCalculator):
    name = "histogram"
    
    def convert(self, arg):
        return arg.apply(str)
    
    def prepare(self, ref):
        
        
        
        ref = self.convert(ref)
        self._ref_counts = Counter(ref)
        refv = np.array(list(self._ref_counts.values()))
        self._ref_norm = dict(zip(ref.keys(), refv/refv.sum() ))
        super().prepare(ref)

    
    def _predict(self, sample):
        sample_counts = Counter(sample)
        keys = sorted(set(sample_counts.keys()).union(self._ref_counts.keys()))
        obs = np.array([sample_counts.get(k, 0) for k in keys])
        obs_norm = obs / obs.sum()
        
        out = {
            "x": list(map(str,keys)),
            "count": obs,
            "probability": obs_norm,
            "intersection": histogram_intersection(sample_counts, self._ref_counts),
            "normalized_intersection": histogram_intersection(dict(zip(keys, obs_norm)), self._ref_norm)
        }

        return out


class NumericalHistIntersectionCalculator(HistIntersectionCalculator):
    
    __min_value = -1e100
    __max_value = 1e100

    def __init__(self,  bins=12, from_ref=True, count_nan=False, include_inf=True,  **kwargs):
        """
        Parameters
        ----------
        bins : number of points to use when creating bins or the bin edges themselves
        from_ref : use the reference data to create the bins
        bins : set bins instead of calculating them from sample or reference
        calc_intersection : add intersection calculation (with ref)
        include_inf : if you want to include -inf and inf in bins
        """
        super().__init__(**kwargs)
        self.bins = bins
        self.from_ref = from_ref
        self.count_nan = count_nan
        self.include_inf = include_inf
        
        self._intervals = None
        
    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")
    
    
    def _get_bin_intervals(self, ref):
                
        if np.ndim(self.bins) == 0:
            if self.include_inf:
                bins = np.hstack(([self.__min_value], np.linspace(ref.min(), ref.max(), self.bins-2), [self.__max_value]))
            else:
                bins = np.linspace(ref.min(), ref.max(), self.bins)
        else:
            bins = self.bins
            
        if np.ndim(bins) == 1:
            bins = list(zip(bins, bins[1:]))
        bins = pd.IntervalIndex([pd.Interval(b[0], b[1]) for b in bins])
        labels = {interval:i for i, interval in enumerate(bins)}
        return bins, labels
            
    def prepare(self, ref):
        ref = self.convert(ref)
        self._intervals, self._labels = self._get_bin_intervals(ref)
        ref = pd.cut(ref, self._intervals)
        ref = ref.apply(lambda x: self._labels.get(x, None))
        return super().prepare(ref)
    
    def _predict(self, sample):
        sample = pd.cut(sample, self._intervals)
        sample = sample.apply(lambda x: self._labels.get(x, None))
        out = super()._predict(sample)
    
        ## Add missing intervals into x, count and density
        d = dict(zip(out.pop('x', []), list(zip(out.pop('count', []), out.pop('density', [])))))
        out['x'] = list(self._labels.values())
        out['count'], out['density'] = zip(*[d.get(str(i), (0,0)) for i in out['x']])
        out["intervals"] = {v:(k.left, k.right) for k,v in sorted(self._labels.items(), key=lambda x: x[1])}
        return out


class KdeIntersectionCalculator(BaseDriftCalculator):
    name = "kde"

    def prepare(self, ref):
        ref = pd.to_numeric(ref, errors="coerce")
        ref = ref[~ref.isnull()]
        super().prepare(ref)

    def __init__(self, npoints=250, method="scott", tol=0.2, **kwargs):
        super().__init__(**kwargs)
        self.npoints = npoints
        self.method = method
        self.tol = tol

    def _predict(self, sample):
        sample = pd.to_numeric(sample, errors="coerce")
        sample = sample[~sample.isnull()]
        if not len(sample): return {}
        if sample.nunique() < 2: return {}
        out = {"intersection_pts": {}}
                
        (out["intersection_density"], 
         out["intersection_pts"]["x"], out["intersection_pts"]["density"], 
         out["x"], _, out["density"]) = kde_intersection_area(
            self._ref, sample, method=self.method, tol=self.tol, npoints=self.npoints)
         
        (out["intersection_probability"], 
         out["intersection_pts"]["x2"], out["intersection_pts"]["probability"], 
         out["x2"], _, out["probability"]) = kde_intersection_area(
            self._ref, sample, method=self.method, tol=self.tol, npoints=self.npoints, use_prob=True)
         
    
        return out

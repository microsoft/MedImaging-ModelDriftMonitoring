import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import colors as mpl_colors
from collections import defaultdict
import pandas as pd


def to_rgba(rgb, alpha=None):
    rgb = mpl_colors.to_rgb(rgb)
    if alpha is None:
        return "rgb(%s, %s, %s)" % (rgb[0], rgb[1], rgb[2])
    return "rgba(%s, %s, %s, %s)" % (rgb[0], rgb[1], rgb[2], alpha)


def line_maker(color, **l):
    return dict(color=color, **l)


def marker_maker(color, **l):  # noqa
    return dict(color=color)


def smooth(y: pd.DataFrame, span=7):
    if span > 0:
        ys = y.ewm(span=span, ignore_na=False).mean()
        ys[y.isna()] = None
    else:
        ys = y
    return ys


def add_date_line(fig, date, name, y=1.08):
    fig.add_shape(type='line',
                  x0=date,
                  y0=0,
                  x1=date,
                  y1=1,
                  line=dict(color='black', dash='dot'),
                  xref='x',
                  yref='paper'
                  )
    fig.add_annotation(textangle=0,
                       xref="x",
                       yref="paper", x=date, y=y,
                       text=name, showarrow=False,
                       font=dict(size=18))


def add_dates(fig, dates, line_y=1.05, include_date=True):
    for name, date in dates.items():
        if not pd.isna(date):
            n = f"{name}<br />({date})" if include_date else name
            add_date_line(fig, date, n, y=line_y)


def collect_corr(y, yp, name, when, weights_name, start_date=None, end_date=None):
    yp = yp.loc[start_date: end_date]
    y = y.loc[start_date: end_date]
    c, cm = yp.corr(y), smooth(yp).corr(smooth(y))
    return {"name": name, "weights_name": weights_name,
            "corr (raw)": c, "corr (smoothed)": cm, "when": when}


class FigureHelper(object):

    def __init__(self, x=None, color_list=px.colors.qualitative.Plotly, dashes=('solid',), smooth_func=smooth,
                 merge_hover=True):
        self.traces = []
        self.error_traces = []
        self.color_list = color_list
        self.line_picker = itertools.cycle(itertools.product(dashes, self.color_list))
        self.lines = defaultdict(lambda: dict(zip(['dash', 'color'], next(self.line_picker))))
        self.names = set()
        self.smooth = smooth_func
        self.x = x
        self.merge_hover = merge_hover

    def set_line(self, key, line=None):
        line = line or {}
        self.lines[key] = self.lines[key]
        self.lines[key].update(line)
        self.lines[key]['color'] = self.lines[key]['color']
        return self.lines[key]

    @staticmethod
    def make_error_traces(x, yu, yl, name, color, alpha):

        # need to remove nans from error traces
        k = ~(yu.isnull() | yl.isnull())
        xe = x[k]
        yl = yl[k]
        yu = yu[k]

        return [go.Scatter(x=xe,
                           y=yu,
                           hoverinfo="skip",
                           showlegend=False,
                           legendgroup=name,
                           name=name,
                           connectgaps=False,
                           line=dict(width=0),
                           ),
                go.Scatter(x=xe,
                           y=yl,
                           fillcolor=to_rgba(color, alpha),
                           fill='tonexty',
                           hoverinfo="skip",
                           showlegend=False,
                           legendgroup=name,
                           name=name,
                           connectgaps=False,
                           line=dict(width=0),
                           )]

    def add_trace(self, y, name, x=None, kind=go.Scatter, color_key=None, row=1, col=1, line=None,
                  std=None, yu=None, yl=None, **trace_kwargs):
        color_key = color_key or name
        trace_kwargs.setdefault('showlegend', name not in self.names)
        self.names.add(name)
        trace_kwargs.setdefault('legendgroup', name)

        line = self.set_line(color_key, line)
        x = x or self.x
        y = y.reindex(x)
        t = kind(x=x, y=y, name=name, **trace_kwargs)
        if not isinstance(t, go.Bar):
            t.line = line_maker(**line)
        else:
            t.marker = marker_maker(**line)

        self.traces.append((row, col, t))

        if std is not None:
            yu = y + std
            yl = y - std

        if yu is not None and yl is not None:
            for t_ in self.make_error_traces(x, yu, yl, name=name, color=line["color"], alpha=0.2):
                self.error_traces.append((row, col, t_))

    def add_bar(self, y, name, color_key=None, row=1, col=1, line=None, include_line=True,
                **trace_kwargs):

        if include_line:
            self.add_trace(y=y, name=name, color_key=color_key, line=line, row=row, col=col, **trace_kwargs)
        self.add_trace(y=y, name=name, color_key=color_key, kind=go.Bar, line=line, row=row, col=col, **trace_kwargs)

    def make_fig(self, **fig_kwargs):

        data = {}
        max_row = 1
        max_col = 1
        for r, c, t in self.traces:
            max_row = max(r, max_row)
            max_col = max(c, max_col)
            data[t.name] = pd.Series(t.y, index=t.x)

        customdata = pd.DataFrame(data)
        fig = make_subplots(rows=max_row, cols=max_col, **fig_kwargs)
        for r, c, t in self.traces:
            if self.merge_hover:
                cus_cols = sorted(customdata)
                ho = "<br />".join(
                    ["{name}=%{{customdata[{i}]:.3f}}".format(i=i, name=name) for i, name in enumerate(cus_cols)])
                hovertemplate = "%{x}<br>" + f"{t.name}: " + "%{y}<br><br>" + f"{ho}<extra></extra>"
                t.customdata = customdata[cus_cols]
                t.hovertemplate = hovertemplate
            fig.add_trace(t, row=r, col=c)

        for r, c, t in self.error_traces:
            fig.add_trace(t, row=r, col=c)
        return fig

import matplotlib
import numpy as np
# Import ordereddict
from collections import OrderedDict
# Import dataclass
from dataclasses import dataclass

@dataclass
class ViolinPlotData:
    vals: list
    col: str
    lab: str    

def violin_plots(axi:matplotlib.axes.Axes, data:OrderedDict[str, ViolinPlotData]) -> matplotlib.axes.Axes:
    # Create a violin plot of the three distributions
    qs = [[0.25, 0.75]] * len(data)
    vals = [d.vals for d in data]
    cols = [d.col for d in data]
    labs = [d.lab for d in data]
    parts = axi.violinplot(vals, showmeans=False, showmedians=True, showextrema=False, quantiles=qs)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cmedians','cquantiles'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    for pos, d in enumerate(vals, start=1):
      q1, q3 = np.percentile(d, [25, 75])
      axi.vlines(pos, q1, q3, color='black', linewidth=1) 
    axi.set_xticks(np.arange(1, len(data)+1))
    axi.set_xticklabels(labs)
    return axi


# For each of the conditions, plot an actual data point (Cin, Cstar, Cest_diag, Cest_free) for a single seed and train. 
# This will check whether the violin plots are representative of the actual data points.
# We'll have one row per condition, and four columns for Cin, Cstar, Cest_diag, Cest_free. 
def scatter_plot(ax, x, y, xlabel, ylabel, c):
    ax.scatter(x.flatten(), y.flatten(), color=c, alpha=0.3)
    ax.set_xlabel(xlabel, labelpad=-2)
    ax.set_ylabel(ylabel)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ll = np.min((xl[0], yl[0]))
    ul = np.max((xl[1], yl[1]))
    lims = [ll, ul]
    ax.plot(lims, lims, ls="--", color="r")
    # 1. Die Markierungen (Zahlenwerte/Ticks) auf die rechte Seite verschieben
    ax.yaxis.tick_right()
    # 2. Die Position des Labels (Text) auf die rechte Seite setzen
    ax.yaxis.set_label_position("right")


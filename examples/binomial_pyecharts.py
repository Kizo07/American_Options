import numpy as np
import matplotlib.pyplot as plt
import time
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

def lcg_uniform_rn(x_0, n):
    a = 7**5
    b = 0
    m = 2**31 - 1
    x = np.empty(n)
    x[0] = (a * x_0 + b) % m
    for i in range(1, n):
        x[i] = (a * x[i - 1] + b) % m
    return x / m

def bernoulli(x_0, p, n):
    x = lcg_uniform_rn(x_0, n)
    return (x < p).astype(int)

def binomial(x_0, n, p, N):
    b = bernoulli(x_0, p, n * N)
    return b.reshape(N, n).sum(axis=1)

x_0 = int(time.time()) % (2**31 - 1)
samples = binomial(x_0, 44, 0.64, 10000)
unique_vals = np.arange(samples.min(), samples.max() + 1)

# Count frequencies for each value
counts = np.bincount(samples, minlength=len(unique_vals))
values = list(range(len(counts)))

# Create Pyecharts Bar chart
bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="1200px", height="600px"))
    .add_xaxis(values)
    .add_yaxis("Frequency", counts.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Binomial Distribution (n=44, p=0.64)"),
        xaxis_opts=opts.AxisOpts(name="Value"),
        yaxis_opts=opts.AxisOpts(name="Frequency"),
        datazoom_opts=[opts.DataZoomOpts()],
    )
)

# Display the chart in notebook
bar.render_notebook()

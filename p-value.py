# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.arange(10, step=1)
n = len(x)
print('x= ', x)
nbins = 3
freq, bins = np.histogram(x, bins=nbins)
print('bin划分:{}, 分成{}个bin'.format(bins, len(bins) - 1))
print('各个bin的频数', freq)
freq_rate = freq / n
print('各个bin的频率', freq_rate)
bin_w = bins[1] - bins[0]
bin_h = freq_rate / bin_w
print('各个bin的高度', bin_h)
# ==================1. 通过distplot
ax = sns.distplot(x, bins=nbins,
                  hist=True,  # Whether to plot a (normed) histogram.
                  kde=True,
                  norm_hist=True,  # norm_hist = norm_hist or kde or (fit is not None); 如果为False且kde=False, 则高度为频数
                  kde_kws={"label": "density_est_by_sns.distplot",
                           "bw": bin_w  # 带宽h, 确保和density_est的h一样, 和kdeplot的bw一样
                           }
                  )
ax.grid(True)
ax.set_yticks(np.arange(0.16, step=0.01))


# ==================2. 通过手动计算density_est
def density_est(x, xs, h):
    """
    给定一组数据xs和带宽h, 计算概率密度函数,
    x: 是f(x)的自变量
    """
    f = 0
    n = len(xs)  # 观测点的个数
    a = 1 / (np.sqrt(2 * np.pi) * n * h)
    for xi in xs:
        f = f + np.exp(-(x - xi) ** 2 / (2 * (h ** 2)))
    f = a * f
    return f


dots = np.linspace(x.min() - bin_w * n * 0.5, x.max() + bin_w * n * 0.5, num=1000)
ax.plot(dots, density_est(dots, x, h=bin_w), c='r', label='density_est_manual')
ax.legend(loc='best')

# ==================3. 通过kdeplot
sns.kdeplot(x, bw=bin_w, ax=ax, label='density_est_by_sns.kdeplot')
ax.plot()

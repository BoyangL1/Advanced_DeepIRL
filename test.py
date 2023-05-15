import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
x = np.random.randn(100)  # 造一个shape为(100,)，服从正态分布的对象x


def histKernel(x):
    plt.figure(dpi=120)
    rc = {'font.sans-serif': 'Times New Roman',
          'axes.unicode_minus': False}
    sns.set_style(style='dark', rc=rc)
    sns.set_style({"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(x,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                              },
                     color='#098154',
                     axlabel='Xlabel',  # 设置x轴标题
                     )
    plt.show()

histKernel(x)
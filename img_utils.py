import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns

def show_img(img):
    print(img.shape, img.dtype)
    plt.imshow(img[:, :, 0])
    plt.ion()
    plt.show()


def heatmap2d(hm_mat, title='', block=False, fig_num=1, text=True):
    """
    Display heatmap
    input:
      hm_mat:   mxn 2d np array
    """
    print('map shape: {}, data type: {}'.format(hm_mat.shape, hm_mat.dtype))

    if block:
        plt.figure(fig_num)
        plt.clf()

    plt.imshow(hm_mat, interpolation='nearest')
    plt.title(title)
    plt.colorbar()

    if text:
        for y in range(hm_mat.shape[0]):
            for x in range(hm_mat.shape[1]):
                plt.text(x, y, '%.1f' % hm_mat[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )

    if block:
        plt.ion()
        print('press enter to continue')
        plt.show()
        plt.waitforbuttonpress()


def heatmap3d(hm_mat, title=''):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    data_2d = hm_mat

    data_array = np.array(data_2d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)

    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]),
                                 np.arange(data_array.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             1, 1, z_data)
    plt.show()
    plt.waitforbuttonpress()


def rewardVisual(rewards, idx_fnid, gpd_file,title="", text=True):
    gdf = gpd.read_file(gpd_file)
    gdf['reward'] = 0
    for i in range(len(rewards)):
        fnid = idx_fnid[i]
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = rewards[i]
    gdf.plot(column='reward', cmap='viridis')
    plt.title(title)


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
    plt.savefig('./kernel.png',dpi=400)
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    feature_map_excel = pd.read_excel('./data/nanshan_tfidf.xlsx')
    states_list = list(feature_map_excel['fnid'])
    fnid_idx = {}
    idx_fnid = {}
    for i in range(len(states_list)):
        fnid_idx.update({states_list[i]: i})
        idx_fnid.update({i: states_list[i]})
    rewards = np.random.randint(100, size=(len(idx_fnid)))
    rewardVisual(rewards, idx_fnid, '')

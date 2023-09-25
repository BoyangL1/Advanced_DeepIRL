from deep_irl_realworld import *
import os
import pandas as pd
import geopandas as gpd

ACT_RAND = 0.3

def readFeatureMap(feature_map_file):
    """
    read feature map from csv feature file

    Args:
        feature_map_file : path of feature file

    Returns:
        feat_map: numpy feature map
        fnid_idx:{fnid:idx}
        idx_fnid:{idx:fnid}
    """
    feature_map_excel = pd.read_csv(feature_map_file)
    states_list = list(feature_map_excel['fnid'])
    fnid_idx = {}
    idx_fnid = {}
    for i in range(len(states_list)):
        fnid_idx.update({states_list[i]: i})
        idx_fnid.update({i: states_list[i]})
    states_list = list(feature_map_excel['fnid'])
    # get feature map of state 
    feat_map = feature_map_excel.iloc[:, 1:] 
    feat_map = np.array(feat_map)
    return feat_map, fnid_idx, idx_fnid, states_list

def testRewardFile(feature_map_file,gpd_file,gpd_save_file,img_save_file,genderAge):
    """
    test reward ckpt file trained by deepirl
    """
    lr = 0.02
    feat_map, _, idx_fnid, states_list = readFeatureMap(feature_map_file)
    nn_r = DeepIRLFC(feat_map.shape[1], lr, genderAge, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return
    
    oh = [genderAge for _ in range(feat_map.shape[0])]
    rewards = normalize(nn_r.get_rewards(feat_map, oh))
    # visualize reward through geopandas
    gdf = gpd.read_file(gpd_file)
    gdf['reward'] = 0
    for i in range(len(rewards)):
        fnid = idx_fnid[i]
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = rewards[i]
    gdf.to_file(gpd_save_file, driver='ESRI Shapefile', crs=4326,encoding='utf-8')
    gdf.plot(column='reward', cmap='viridis', legend=False)
    plt.title('reward recovered from ckpt file')
    plt.savefig(img_save_file, dpi=600)
    plt.show()


if __name__ == "__main__":
    genderAge=[1,0,0,0,1]
    feature_map_file="./data/sz_sf.csv"
    gpd_file="./data/ss_city_grid.shp"
    gpd_save_file="./data/ss_reward"
    img_save_file="./img/ss_reward.png"
    testRewardFile(feature_map_file,gpd_file,gpd_save_file,img_save_file,genderAge)
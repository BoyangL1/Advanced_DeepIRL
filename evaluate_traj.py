import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from deep_irl_realworld import *
from test_dirl_realworld import readFeatureMap
from A_star.A_star_traj import *
from DTW.dtw import DTW

def getRewardFile(feat_map):
    """
    get reward of feature map

    Args:
        feat_map (N*M:2d_matrix): feature map of zone

    Returns:
        rewards (N*1): reward of each grid
    """
    nn_r = DeepIRLFC(feat_map.shape[1], 0.02, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return

    rewards = normalize(nn_r.get_rewards(feat_map))
    return rewards

def trajContrast(route, real_map, gpd_file,if_show):
    """
    Comparison of real and generated trajectories

    Args:
        routes_file 
        real_map : real map class
        gpd_file 
    """    
    start_fnid = route[0]
    end_fnid = route[-1]

    route_generate = astar(real_map, start_fnid, end_fnid)

    gdf = gpd.read_file(gpd_file)
    ColNames = gdf.columns
    route_df = pd.DataFrame(columns=ColNames)
    route_generate_df = pd.DataFrame(columns=ColNames)

    for fnid in route:
        idx = gdf[(gdf['fnid'] == fnid)].index
        route_df = route_df.append(gdf.iloc[idx, :], ignore_index=True)
    for fnid in route_generate:
        idx = gdf[(gdf['fnid'] == fnid)].index
        route_generate_df = route_generate_df.append(
            gdf.iloc[idx, :], ignore_index=True)
    route_generate_gdf = gpd.GeoDataFrame(
        route_generate_df, geometry="geometry")
    route_gdf = gpd.GeoDataFrame(
        route_df, geometry="geometry")

    plt.rcParams["font.family"] = "Times New Roman"
    _, ax = plt.subplots()
    ax=gdf.plot(ax=ax, color='darkgray', label='map')
    ax=route_gdf.plot(ax=ax, color='lightsalmon',label = "real trajecory")
    ax=route_generate_gdf.plot(ax=ax, color='lightskyblue',label = "generated trajecory")
    ax.axis('off')
    plt.title('trajectory contrast')
    if if_show:
        plt.show()

def trajEvaluate(routes_file,real_map,gpd_file,save_picture=False):
    routes = np.load(routes_file)
    all_length=0
    list_length=[]
    for i,route in enumerate(routes):
        # generate trajectory
        start_fnid=route[0]
        end_fnid = route[-1]
        route_generate = astar(real_map, start_fnid, end_fnid)
        # calculate dtw distance between trajectory
        traj_dtw=DTW(route,route_generate)
        D=traj_dtw.dtw()
        length=(len(route_generate)+len(route))/2
        dtw_length=D[-1][-1]/length
        # save figure
        if save_picture:
            trajContrast(route, real_map, gpd_file, False)
            plt.title('DTW Length is {}'.format(dtw_length))
            plt.savefig('./img/traj_distance/{}.png'.format(i))
        print(dtw_length)
        all_length+=dtw_length
        list_length.append(dtw_length)

    avg_length=all_length/len(routes)
    return avg_length,list_length

if __name__ == "__main__":
    feat_map_file = './data/poor_zone_tfidf.xlsx'
    feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(feat_map_file)
    rewards = getRewardFile(feat_map).tolist()
    cost = [1-r[0] for r in rewards]

    real_map = Map(states_list, fnid_idx, idx_fnid, cost*10)

    gpd_file = './data/poor_zone.shp'
    routes_file = './data/routes_states_poor.npy'

    # Trjectory comparison
    routes = np.load(routes_file)
    route_idx = np.random.randint(0, len(routes)-1)
    route = routes[route_idx]
    trajContrast(route, real_map, gpd_file, True)

    # # trajectory evaluation
    # avg_length,list_length=trajEvaluate(routes_file,real_map,gpd_file)
    # print(avg_length)
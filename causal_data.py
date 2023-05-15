from deep_irl_realworld import *
import pandas as pd

def getReward(feature_map_file,gpd_file,model_file):
    """
    get reward map from ckpt file
    """
    lr = 0.02
    
    # get feature map of state
    feature_map_excel = pd.read_excel(feature_map_file)
    feature_map1 = feature_map_excel.iloc[:, 2:11]
    feature_map2 = feature_map_excel.iloc[:, 11:]
    feat_map_df = pd.concat([feature_map1, feature_map2], axis=1)
    feat_map = np.array(feat_map_df)

    nn_r = DeepIRLFC(feat_map.shape[1], lr, 40, 30)
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return
    rewards = normalize(nn_r.get_rewards(feat_map))

    return rewards,feat_map_df

if __name__=="__main__":
    feature_map_file = './data/poor_zone_tfidf.xlsx'
    gpd_file = './data/poor_zone.shp'
    model_file = './model'
    reward,feat_df=getReward(feature_map_file,gpd_file,model_file)
    reward_df=pd.DataFrame(reward)
    feat_reward_df=pd.concat([feat_df,reward_df],axis=1)
    feat_reward_df.to_csv('./data/poor_feat_reward.csv',index=False,header=None)
from deep_irl_realworld import *
import pandas as pd


def getReward(feature_map_file, genderAge, model_file):
    """
    get reward map from ckpt file
    """
    lr = 0.02

    # get feature map of state
    feature_map_excel = pd.read_csv(feature_map_file)
    feat_map_df = feature_map_excel.iloc[:, 1:]
    feat_map = np.array(feat_map_df)

    nn_r = DeepIRLFC(feat_map.shape[1], lr, genderAge, 40, 30)
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return

    oh = [genderAge for _ in range(feat_map.shape[0])]
    rewards = normalize(nn_r.get_rewards(feat_map, oh))

    return rewards, feat_map_df


if __name__ == "__main__":
    feature_map_file = './data/ns_sf.csv'
    gpd_file = './data/nanshan_grid.shp'
    model_file = './model'
    route_file_path = './data/routes_states'

    final_df = pd.DataFrame()
    for f in os.listdir(route_file_path):
        genderAge = [0]*5
        gender, age = int(f[0]), int(f[2])
        genderAge[gender], genderAge[age+2] = 1, 1
        tf.reset_default_graph()
        reward, feat_df = getReward(feature_map_file, genderAge, model_file)
        reward = [r[0] for r in reward]

        reward_df = pd.DataFrame({'reward': reward})
        df_length = len(reward_df)
        gender_age_df = pd.DataFrame({'male': [genderAge[0]]*df_length, 'female':[genderAge[1]]*df_length,
                                     'young': [genderAge[2]]*df_length, 'middle': [genderAge[3]]*df_length, 'old': [genderAge[4]]*df_length})
        feat_reward_df = pd.concat([feat_df, gender_age_df, reward_df], axis=1)
        final_df = final_df.append(feat_reward_df, ignore_index=True)
    final_df.to_csv('./data/nanshan_reward.csv',index=False,header=None)
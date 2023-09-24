import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple

dirs = {0: 1, 1: -1, 2: -357, 3: 357, 4: 0}
Step=namedtuple('Step',['state','action'])

def trajFromPolicyFile(policys, fnid_idx, start_fnid, traj_len):
    traj = [start_fnid]
    start_fnid_idx = fnid_idx[start_fnid]

    for i in range(traj_len):
        this_action = policys[start_fnid_idx]
        this_fnid = start_fnid+dirs[this_action]
        traj.append(this_fnid)
        start_fnid = this_fnid
    return traj

def trajFromExpert(states_tuple):
    state_action = np.load(states_tuple, allow_pickle=True)

    num_trajectories = len(state_action)
    num_trajectories_to_select = num_trajectories // 10  # Select one-tenth of the trajectories

    selected_indices = np.random.choice(num_trajectories, num_trajectories_to_select, replace=False)

    traj = []
    for idx in selected_indices:
        route = state_action[idx]
        one_traj = []
        for r in route:
            one_traj.append(r.state)
        traj.append(one_traj)

    return traj

def logLikelihood(df1,df2,column):
    column1_df1 = df1[column]
    column1_df2 = df2[column]
    mean_df1, std_dev_df1 = column1_df1.mean(), column1_df1.std()
    mean_df2, std_dev_df2 = column1_df2.mean(), column1_df2.std()
    
    def log_likelihood_ratio(x, mean1, std_dev1, mean2, std_dev2):
        pdf1 = norm.pdf(x, loc=mean1, scale=std_dev1)
        pdf2 = norm.pdf(x, loc=mean2, scale=std_dev2)
        return np.log(pdf1 / pdf2)

    log_likelihoods_df1 = column1_df1.apply(lambda x: log_likelihood_ratio(x, mean_df1, std_dev_df1, mean_df2, std_dev_df2))
    log_likelihoods_df2 = column1_df2.apply(lambda x: log_likelihood_ratio(x, mean_df1, std_dev_df1, mean_df2, std_dev_df2))

    log_likelihood_mean_df1 = log_likelihoods_df1.mean()
    log_likelihood_mean_df2 = log_likelihoods_df2.mean()

    return abs(log_likelihood_mean_df1-log_likelihood_mean_df2)

def trajLogLikelihood(feature_map_file, expert_traj_list, generated_traj_list):
    feature_map_excel = pd.read_csv(feature_map_file)

    e_fnid = list(set(expert_traj_list))
    g_fnid = list(set(generated_traj_list))

    e_df = feature_map_excel[feature_map_excel['fnid'].isin(e_fnid)]
    g_df = feature_map_excel[feature_map_excel['fnid'].isin(g_fnid)]

    column_names = feature_map_excel.columns.tolist()[1:31]

    log_likelihood = []

    for n in column_names:
        log_likelihood.append(logLikelihood(e_df,g_df,n))

    # Filter out inf and nan values
    log_likelihood = [x for x in log_likelihood if not (np.isinf(x) or np.isnan(x))]
    return np.nanmean(log_likelihood)        
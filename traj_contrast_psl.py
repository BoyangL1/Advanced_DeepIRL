from collections import namedtuple
from dnn_psl import *

Step=namedtuple('Step',['state','action'])

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

# load model
loaded_model = DNNPSLModel(35)
loaded_model.load_state_dict(torch.load('./model/dnn psl/model_weights.pth'))
loaded_model.eval()
# load grid network
graphml_file = "./data/nanshan_network.graphml"

# load feature map
feature_map_file="./data/ns_sf.csv"
df = pd.read_csv(feature_map_file)
# load feature map
feat_map_file = './data/ns_sf.csv'
feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(feat_map_file)


origin, destination = 9719.0, 11848.0
G=nx.read_graphml(graphml_file)
paths=nShortestPaths(G, origin, destination, num_paths_to_find=10)

# path features
path_features=[]
for path in paths:
    filtered_rows = df[df['fnid'].isin(path)]
    feature = filtered_rows[1:]
    path_features.append(filtered_rows.iloc[:, 1:].values.tolist())
column_means = [np.mean(np.array(inner_list).T, axis=1) for inner_list in path_features]
column_means = np.array(column_means)
genderAge = [1,0,1,0,0]
genderAge = np.array(genderAge)
column_means = np.hstack((column_means, np.tile(genderAge, (len(column_means), 1))))
X = torch.tensor(column_means, dtype=torch.float32)

outputs = loaded_model(X)
generated_traj = paths[np.argmax(outputs.detach().numpy())]

print(generated_traj)
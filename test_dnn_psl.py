from dnn_psl import *

loaded_model = DNNPSLModel(35)
loaded_model.load_state_dict(torch.load('./model/dnn psl/model_weights.pth'))
loaded_model.eval()

graphml_file = "./data/nanshan_network.graphml"
G=nx.read_graphml(graphml_file)

state_action = np.load('./data/routes_states/0_0_states_tuple.npy',allow_pickle=True)
selected_traj = np.random.choice(state_action, size=1, replace=False)[0]
origin, destination = int(selected_traj[0].state), int(selected_traj[-1].state)
paths=nShortestPaths(G, origin, destination, num_paths_to_find=10)
selected_traj = [t.state for t in selected_traj]
paths.append(selected_traj)

# path features
path_features=[]
feature_map_file="./data/ns_sf.csv"
df = pd.read_csv(feature_map_file)
for path in paths:
    filtered_rows = df[df['fnid'].isin(path)]
    feature = filtered_rows[1:]
    path_features.append(filtered_rows.iloc[:, 1:].values.tolist())
column_means = [np.mean(np.array(inner_list).T, axis=1) for inner_list in path_features]
# Convert the lists to NumPy arrays
column_means = np.array(column_means)
genderAge = [1,0,1,0,0]
genderAge = np.array(genderAge)
# Stack genderAge horizontally to each row in column_means
column_means = np.hstack((column_means, np.tile(genderAge, (len(column_means), 1))))
# Convert path features to tensors
X = torch.tensor(column_means, dtype=torch.float32)

outputs = loaded_model(X)
print("the max logit is {}".format(np.argmax(outputs.detach().numpy())+1))
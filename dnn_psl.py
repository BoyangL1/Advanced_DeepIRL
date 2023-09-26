import networkx as nx
import numpy as np
from collections import namedtuple
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Step=namedtuple('Step',['state','action'])

def nShortestPaths(G, origin, destination, num_paths_to_find=10):
    origin_node = [(node, data) for node, data in G.nodes(data=True) if data.get('fnid') == origin]
    destination_node = [(node, data) for node, data in G.nodes(data=True) if data.get('fnid') == destination]

    if not origin_node or not destination_node:
        print("Origin or destination nodes not found.")
        return

    origin = origin_node[0][0]
    destination = destination_node[0][0]

    shortest_paths = []
    for _ in range(num_paths_to_find):
        shortest_path = nx.shortest_path(G, source=origin, target=destination)
        shortest_paths.append(shortest_path)

        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            G.remove_edge(u, v)

    paths = []
    for i, path in enumerate(shortest_paths):
        fnid_values = [G.nodes[node]['fnid'] for node in path]
        paths.append(fnid_values)

    return paths

class DNNPSLModel(nn.Module):
    def __init__(self, input_size):
        super(DNNPSLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  # Output layer with one neuron for logits

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    route_file_path = './data/routes_states'
    graphml_file = "./data/nanshan_network.graphml"

    # Instantiate your model and define the loss function
    input_size = 35  # Match the input size to your feature dimension
    model = DNNPSLModel(input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Define an optimizer (e.g., SGD or Adam)

    while True:
        for f in os.listdir(route_file_path):
            genderAge = [0]*5
            gender,age=int(f[0]),int(f[2])
            genderAge[gender], genderAge[age+2] = 1,1
            state_action = np.load(route_file_path+'/'+f,allow_pickle=True)
            selected_traj = np.random.choice(state_action, size=1, replace=False)[0]
            origin, destination = int(selected_traj[0].state), int(selected_traj[-1].state)

            G=nx.read_graphml(graphml_file)
            try:
                paths=nShortestPaths(G, origin, destination, num_paths_to_find=10)
            except:
                print("the shortest path don't exit")
                continue
            # add true trajectory
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
            genderAge = np.array(genderAge)
            # Stack genderAge horizontally to each row in column_means
            column_means = np.hstack((column_means, np.tile(genderAge, (len(column_means), 1))))
            # Convert path features to tensors
            X = torch.tensor(column_means, dtype=torch.float32)
            # Create labels (y)
            y = torch.zeros(len(paths))
            y[-1] = 1  # True path has a label of 1, others are 0

            # Training loop
            num_epochs = 1000
            for epoch in range(num_epochs):
                outputs = model(X)
                loss = criterion(outputs, y.unsqueeze(1))  # Ensure the shape matches
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print the loss for monitoring
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

        if loss.item() < 0.001:
            print("Training completed, end the training loop!")
            torch.save(model.state_dict(), './model/dnn psl/model_weights.pth')
            break
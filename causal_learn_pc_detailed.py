import itertools
import math
from itertools import chain, combinations
from pyexpat import model

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr


def dfs(graph, k: int, path: list, vis: list, labels: list):
    """
    Depth First Search for causal relation search

    Args:
        graph : cdpag
        k : start index 
        path : depth first search path 
        vis : if visited
        labels: labels list
    """
    flag = True

    for i in range(len(graph)):
        if (graph[i][k]) and (vis[i] != True):
            flag = True
            vis[i] = True
            path.append(labels[i])
            dfs(graph, i, path, vis, labels)
            path.pop()
            vis[i] = False

    if flag:
        print(path)


def causalGraphPlot(graph, labels: list, pic_path: str, model_path: str):
    """
    visualize beysian network

    Args:
        graph : networkx graph
        labels (list): label list
        path (str): picture save path
        model_path (str): model save path
    """
    G = nx.DiGraph()

    for i in range(len(graph)):
        if i <= 29:
            G.add_node(labels[i], partition=1)
        elif 29 < i < 35:
            G.add_node(labels[i], partition=2)
        else:
            G.add_node(labels[i], partition=3)
        for j in range(len(graph[i])):
            if graph[i][j]:
                # G.add_edges_from([(labels[i], labels[j])])
                G.add_weighted_edges_from(
                    [(labels[i], labels[j], graph[i][j])], weight='weight')

    nx.write_gexf(G, model_path)
    nx.draw(G, with_labels=True)
    # plt.savefig(pic_path)
    plt.show()


def subset(iterable):
    """
    calculate sunbet
    subset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Args:
        iterable : Iterable variables

    Returns:
        iterable : Iterable subset variables
    """
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def skeleton(suffStat, indepTest, alpha, labels, m_max):
    """
    construct skeleton for beysian network

    Args:
        suffStat : sufficient states {"C":correlation coefficient,"n":number of states}
        indepTest : gaussian independence test
        alpha : minimum for conditional independence test
        labels : label list
        m_max : max subset length

    Returns:
        sk+sepset: {"sk":np.array(G),"sepset set":d-seperation set for each point pair}
    """
    # Completely undirected graphs
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]
    G = [[1 for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        G[i][i] = 0

    done = False  # flag

    ord = 0  # subset length
    while done != True and any(G) and ord <= m_max:
        done = True

        # neighboring point set
        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j]:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y]:
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = 0

                # adj(C,x) \ {y}
                neighbors = [i for i in range(
                    len(neighborsBool)) if neighborsBool[i]]

                if len(neighbors) >= ord:

                    # |adj(C, x) \ {y}| > ord
                    if len(neighbors) > ord:
                        done = False

                    # |adj(C, x) \ {y}| = ord
                    for neighbors_S in set(itertools.combinations(neighbors, ord)):
                        # if x and y are d-seperated by neighbors_S
                        # conditional independence,return p-value
                        pval, correlation_r = indepTest(
                            suffStat, x, y, list(neighbors_S))

                        if pval >= alpha:
                            # if pval>alpha, x is independent from y
                            G[x][y] = G[y][x] = 0

                            # add neighbors_S into seperation set
                            sepset[x][y] = list(neighbors_S)
                            break
                        else:
                            G[x][y] = G[y][x] = correlation_r

        ord += 1
    return {'sk': np.array(G), 'sepset': sepset}


def extendCpdag(graph):
    """
    turn a partially directed acyclic graph to completed partially directed acyclic graph

    Args:
        graph : {"skeleton":np.array(G),"seperation set":d-seperation set for each point pair}
    """

    def rule1(pdag):
        """
        If there is a chain a -> b - c, and a, c is not adjacent, change b - c to b - > c

        Args:
            pdag : partially directed acyclic graph

        Returns:
            pdag: partially directed acyclic graph
        """
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] and pdag[j][i] == 0:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] and search_pdag[i][b]) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b][c] and pdag[c][b]:
                        pdag[b][c] = graph['sk'][b][c]
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b]:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag):
        """
        If there is a chain a -> c -> b, change a - b to a -> b

        Args:
            pdag : partially directed acyclic graph

        Returns:
            pdag: partially directed acyclic graph
        """
        search_pdag = pdag.copy()
        ind = []

        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] and pdag[j][i]:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] and search_pdag[i][a] == 0) and (search_pdag[i][b] and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] and pdag[b][a]:
                    pdag[a][b] = graph['sk'][a][b]
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a]:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag):
        """
        If a - c1 - > b and a - c2 - > b, and c1, c2 are not adjacent, change a - b to a -> b

        Args:
            pdag (_type_): _description_

        Returns:
            pdag: partially directed acyclic graph
        """
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] and pdag[j][i]:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] and search_pdag[i][a]) and (search_pdag[i][b] and search_pdag[b][i] == 0):
                    isC.append(i)

            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        # unfaithful
                        if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] and search_pdag[b][a]:
                            pdag[a][b] = graph['sk'][a][b]
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a]:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    pdag = [[0 if graph['sk'][i][j] == 0 else graph['sk'][i][j] for i in range(
        len(graph['sk']))] for j in range(len(graph['sk']))]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j]:
                ind.append((i, j))

    # Change x - y - z to x -> y <- z
    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] and z != x:
                allZ.append(z)

        for z in allZ:
            if graph['sk'][x][z] == 0 and graph['sepset'][x][z] != None and graph['sepset'][z][x] != None and not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[y][x] = pdag[y][z] = 0
                pdag[x][y] = graph['sk'][x][y]
                pdag[z][y] = graph['sk'][z][y]

    # # apply rule1 - rule3
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest, skeleton_path, m_max=float("inf"), verbose=False):
    """
    PC algorithm

    Args:
        suffStat : sufficient states {"C":correlation coefficient,"n":number of states}
        alpha : minimun conditional independence score
        labels : labels of each state
        indepTest : gaussian independence test
        m_max : max sunbset length
        verbose : log display

    Returns:
        cpdag: Completed partially directed acyclic graph
    """
    # skeleton
    graphDict = skeleton(suffStat, indepTest, alpha, labels, m_max)
    df_graph = pd.DataFrame(graphDict['sk'])
    df_graph.to_csv(skeleton_path, index=False)
    # exteng to  CPDAG
    cpdag = extendCpdag(graphDict)
    # print beysian network matrix
    if verbose:
        print(cpdag)
    return cpdag


def gaussCiTest(suffstat, x, y, S):
    """
     test for conditional independence

    Args:
        suffStat : sufficient states {"C":correlation coefficient,"n":number of states}
        x : causal parameter
        y : causal parameter
        S : d-seperation point list

    Returns:
        p-value: conditional independence parameter
    """
    C = suffstat["C"]
    n = suffstat["n"]

    cut_at = 0.9999999

    # Zero-order partial correlation coefficient
    if len(S) == 0:
        r = C[x, y]

    # First-order partial correlation coefficient
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / \
            math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))

    # High-order partial correlation coefficient
    else:
        m = C[np.ix_([x]+[y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1 * cut_at, r))

    # Fisher’s z-transform
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))

    # Φ^{-1}(1-α/2)
    return 2 * (1 - norm.cdf(abs(res))), r


if __name__ == '__main__':
    file_path = './data/nanshan_reward.csv'
    image_path = './img/causalGraph.png'
    model_path = './data/causalGraph.gexf'
    skeleton_path = './data/skeleton.csv'

    data = pd.read_csv(file_path)
    labels = [
        'Population density',
        'Land Use Mix',
        'Open Space Ratio',
        'Intersections',
        'Center',
        'Airport',
        'Railway',
        'Dock',
        'Coach',
        'Expressway',
        'Cycleway',
        'Suburban Road',
        'City Branch',
        'Inner Road',
        'Main Road',
        'Not Built Road',
        'SideWalk',
        'Urban Secondary',
        'Attractions',
        'Food&Beverages',
        'Tranportation',
        'Sport',
        'Public',
        'Enterprises',
        'Medical',
        'Government',
        'Life',
        'Education&Science',
        'Shopping',
        'Finance',
        'Male',
        'Female',
        'Young',
        'Middle',
        'Old',
        'Reward'
    ]

    row_count = len(labels)
    graph = pc(
        suffStat={"C": data.corr().values, "n": data.values.shape[0]},
        alpha=0.05,
        skeleton_path=skeleton_path,
        labels=[str(i) for i in range(row_count)],
        indepTest=gaussCiTest,
        verbose=True
    )

    df_graph = pd.DataFrame(graph)
    df_graph.to_csv('./data/edge_weight.csv', index=False)

    start = -1  # index for 'reward' label
    vis = [0 for i in range(row_count)]
    vis[start] = True
    path = []
    path.append(labels[start])
    dfs(graph, start, path, vis, labels)

    causalGraphPlot(graph, labels, image_path, model_path)

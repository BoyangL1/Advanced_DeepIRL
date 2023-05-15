import sys
import math
class DTW(object):
    def __init__(self,traj,traj_generate):
        self.traj=traj
        self.traj_generate=traj_generate

    def calDistance(self,x, y):
        return math.sqrt(pow(x[0]-y[0],2)+pow(x[1]-y[1],2))

    def dtw(self):
        X=self.fnidToXY(self.traj)
        Y=self.fnidToXY(self.traj_generate)
        l1 = len(X)
        l2 = len(Y)
        D = [[0 for i in range(l1 + 1)] for i in range(l2 + 1)]
        # D[0][0] = 0
        for i in range(1, l1 + 1):
            D[0][i] = sys.maxsize
        for j in range(1, l2 + 1):
            D[j][0] = sys.maxsize
        for j in range(1, l2 + 1):
            for i in range(1, l1 + 1):
                D[j][i] = self.calDistance(X[i - 1], Y[j-1]) + \
                    min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1])
        return D

    def fnidToXY(self,traj):
        traj_list=[]
        for t in traj:
            x=t%357
            y=(t)//357
            traj_list.append((x,y))
        return traj_list

    
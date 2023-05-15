import math
class Map(object):
    def __init__(self,states_list,fnid_idx,idx_fnid,cost):
        self.states = states_list
        self.fnid_idx=fnid_idx
        self.idx_fnid=idx_fnid
        self.cost=cost

# Manhattan Distance
class Node(object):
    def __init__(self,state,g,h,cost,father):
        self.s = state
        self.g = g
        self.h = h
        self.c=cost
        self.father = father

    def calDist(self,s_state,e_state):
        s_x,s_y=s_state%357,s_state//357
        e_x,e_y=e_state%357,e_state//357
        return (abs(s_x-e_x)+abs(s_y-e_y))*10

    def getNeighbor(self,mapState,end_state):
        s =self.s
        result = []

        def calCost(s_state):
            nonlocal mapState
            idx=mapState.fnid_idx[s_state]
            return mapState.cost[idx]
    #up
        if (s+357) in mapState.states:
            start_state=s+357
            upNode = Node(start_state,self.g+10,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #down
        if (s-357) in mapState.states:
            start_state=s-357
            upNode = Node(start_state,self.g+10,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #left
        if (s-1) in mapState.states:
            start_state=s-1
            upNode = Node(start_state,self.g+10,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #right
        if (s+1) in mapState.states:
            start_state=s+1
            upNode = Node(start_state,self.g+10,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #up-left
        if (s+356) in mapState.states:
            start_state=s+356
            upNode = Node(start_state,self.g+14,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #up-right
        if (s+358) in mapState.states:
            start_state=s+358
            upNode = Node(start_state,self.g+14,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #down-left
        if (s-358) in mapState.states:
            start_state=s-358
            upNode = Node(start_state,self.g+14,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)
    #down-right
        if (s-356) in mapState.states:
            start_state=s-356
            upNode = Node(start_state,self.g+14,self.calDist(start_state,end_state),calCost(start_state),self)
            result.append(upNode)

        return result

    def hasNode(self,worklist):
        for i in worklist:
            if(i.s==self.s):
                return True
        return False

    # if hasNode=True
    def changeG(self,worklist):
        for i in worklist:
            if(i.s==self.s):
                if(i.g>self.g):
                    i.g = self.g

def getKeyforSort(element:Node):
    return element.g+element.c #element, do not plus element.h
    
def astar(workMap,start_state,end_state):
    startNode = Node(start_state, 0, 0, 0, None)
    openList = []
    lockList = []
    lockList.append(startNode)
    currNode = startNode

    while end_state != currNode.s:
        workList = currNode.getNeighbor(workMap,end_state)
        for i in workList:
            if (i not in lockList):
                if(i.hasNode(openList)):
                    i.changeG(openList)
                else:
                    openList.append(i)
        openList.sort(key=getKeyforSort) 
        currNode = openList.pop(0)
        lockList.append(currNode)
    
    result = []
    while(currNode.father!=None):
        result.append((currNode.s))
        currNode = currNode.father
    result.append((currNode.s))
    return result
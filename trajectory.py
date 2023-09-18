from collections import namedtuple
import geopandas as gpd
import matplotlib.pyplot as plot
from shapely import geometry
import pandas as pd
import itertools
from collections import deque
import numpy as np

actions = [0, 1, 2, 3, 4]
dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
#        right,    left,   down,   up ,   stay
Step=namedtuple('Step',['state','action'])

def getActionOfStates(route_state):
    state_action=[]
    length=len(route_state)
    first_state=route_state[0]
    if length==1:
        step=Step(state=first_state,action=4)
        state_action.append(step)
        return state_action
    for i in range(1,length):
        second_state=route_state[i]

        def getAction(first_s,second_s):
            if second_s-first_s==1:
                return 0
            elif second_s- first_s==-1:
                return 1
            elif second_s-first_s==357:
                return 3
            elif second_s-first_s==-357:
                return 2

        idx_minux=second_state-first_state
        if idx_minux==358:
            second_state=second_state-1
            act=getAction(first_state,second_state)
            step=Step(state=first_state,action=act)
            state_action.append(step)
            first_state=second_state
            second_state+=1
        elif idx_minux==356:
            second_state=second_state+1
            act=getAction(first_state,second_state)
            step=Step(state=first_state,action=act)
            state_action.append(step)
            first_state=second_state
            second_state-=1
        elif idx_minux==-358:
            second_state=second_state+1
            act=getAction(first_state,second_state)
            step=Step(state=first_state,action=act)
            state_action.append(step)
            first_state=second_state
            second_state-=1
        elif idx_minux==-356:
            second_state=second_state-1
            act=getAction(first_state,second_state)
            step=Step(state=first_state,action=act)
            state_action.append(step)
            first_state=second_state
            second_state+=1
        act=getAction(first_state,second_state)
        step=Step(state=first_state,action=act)
        state_action.append(step)
        first_state=second_state
    step=Step(state=second_state,action=4)
    state_action.append(step)
    return state_action

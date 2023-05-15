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
Step = namedtuple('Step', ['state', 'action'])


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


def getFnidByPoint(first_point, second_point):
    """
    return the fnid of each line

    Args:
        first_point : start position of line
        second_point : end position of line

    Returns:
        fnid_list: fnid list
    """
    fnid_list = []
    geometry_first_point = gpd.GeoSeries(geometry.Point(first_point), crs=4326)
    geometry_second_point = gpd.GeoSeries(
        geometry.Point(second_point), crs=4326)
    first_point_ = gpd.GeoDataFrame(geometry=geometry_first_point, crs=4326)
    second_point_ = gpd.GeoDataFrame(geometry=geometry_second_point, crs=4326)
    intersect_first = gpd.overlay(
        df1=district, df2=first_point_, how='intersection', keep_geom_type=False)
    fnid_list.extend(list(intersect_first['fnid']))
    # if distance>250m,then interpolation,
    # the projection coordinate system of ShenZhen is UTM 49N(EPSG:3406)
    UTM_first_point = geometry_first_point.to_crs(3406)
    UTM_second_point = geometry_second_point.to_crs(3406)
    dist = UTM_first_point.distance(UTM_second_point)
    delta = int(dist//50)
    for i in range(delta):
        interpolation_x = first_point[0]+i * \
            (second_point[0]-first_point[0])/delta
        interpolation_y = first_point[1]+i * \
            (second_point[1]-first_point[1])/delta
        geometry_interpolation = gpd.GeoSeries(geometry.Point(
            [interpolation_x, interpolation_y]), crs=4326)
        interpolation_point = gpd.GeoDataFrame(
            geometry=geometry_interpolation, crs=4326)
        intersect_ = gpd.overlay(
            df1=district, df2=interpolation_point, how='intersection', keep_geom_type=False)
        fnid_list.extend(list(intersect_['fnid']))
    intersect_second = gpd.overlay(
        df1=district, df2=second_point_, how='intersection', keep_geom_type=False)
    fnid_list.extend(list(intersect_second['fnid']))
    return fnid_list


if __name__ == "__main__":
    routes = gpd.read_file('./data/route1.shp')
    district = gpd.read_file('./data/nanshan.shp')
    # get states of each trajectory
    length = len(routes)
    routes_states = []
    for i in range(length):
        geometry_ = routes.iloc[i, -1]
        q = deque(geometry_.coords)
        first_point = q.popleft()
        states = []
        states_unique = []
        while q:
            second_point = q.popleft()
            fnid_list = getFnidByPoint(first_point, second_point)
            if fnid_list:
                states.extend(fnid_list)
            first_point = second_point
        for key, _ in itertools.groupby(states):
            states_unique.append(key)
        print("the {} route finished".format(i))
        routes_states.append(states_unique)
    routes_states = np.array(routes_states)
    np.save('./data/routes_states.npy', routes_states)
    print(np.load('./data/routes_states.npy', allow_pickle=True) == routes_states)
    # get action of each state
    state_action_tuple = []
    state_action_tuple=[]
    for route_state in routes_states:
        sta_act=getActionOfStates(route_state)
        state_action_tuple.append(sta_act)
    state_action_tuple = np.array(state_action_tuple)
    np.save('./data/state_action_tuple.npy', state_action_tuple)
    print(np.load('./data/state_action_tuple.npy',
          allow_pickle=True) == state_action_tuple)

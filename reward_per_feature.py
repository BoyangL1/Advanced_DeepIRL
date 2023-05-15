from deep_irl_realworld import *
import os
import pandas as pd


if __name__ == "__main__":
    data=[[0]*30 for _ in range(100*30)]
    val=[]
    for j in range(100):
        val.append(j*0.01)

    for i in range(100*30):
        data[i][int(i//100)]=val[i%100]
    feat_map=pd.DataFrame(data=data)
    nn_r = DeepIRLFC(feat_map.shape[1], 0.02, 40, 30)
    # model_file = './model/poor'
    model_file = './model/villa'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
    rewards = normalize(nn_r.get_rewards(feat_map))
    plt.figure(figsize=(15, 8), dpi=150)
    plt.rc('font',family='Times New Roman')
    for i in range(30):
        r=rewards[i*100:(i+1)*100]
        x=[i for i in range(i*100,(i+1)*100)]
        
        plt.vlines(x[-1], 0, r[-1], linestyles='dashed', colors='red',alpha=0.3)
        plt.vlines(x[0], 0, r[0], linestyles='dashed', colors='red',alpha=0.3)

        plt.plot(x,r)
    labels = ['MainRoad','InnerRoad','CityBranch','UrbanSecondary','SuburbanRoad','SideWalks','FreeWay','CycleWay','UnBuiltRoad','Transportation'
    ,'Accommodation','Sport','Public','Enterprises','Medical','Commercial','Indoor','Motorcycle','Government','CarService','CarRepair','CarSale',
    'Life','Science&Education','Shopping','PassFacilities','RoadFurniture','Finance','Attractions','Food&Beverages']
    x=[(i+0.5)*100 for i in range(30)]
    plt.xticks(x,labels,rotation=45)
    plt.tick_params(labelsize=8)
    plt.title('Contribution of a single feature to the reward value',fontsize=15)
    plt.ylabel('REWARD',fontsize=15)
    plt.show()
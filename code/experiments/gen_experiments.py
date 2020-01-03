import json
import os
import math

experiment_name = 'all_sa'

networks = ['PointNet','GNN','PointNet++'][:1]
samplings = ['random','furthest']#[:1]
datashares = [0.1,0.25,0.5,0.75,1.0]#[-1:]
datatypes = ['xyz','rgb','nxyz'][:1]
architectures = ['FullSplit','HalfSplit','NoSplit'][2:]
loss_weights = [0.5,0.75,0.8,0.9,0.95,0.98,0.99,0.995][5:6]
optimizers = ['sgd','adam','ranger','ralamb'][2:3]
lrs = [0.01,0.005,0.001][1:2]
nneighbors = [5,10,15][1:2]
#rneighbors = [[0.03,0.06],[0.03,0.06]][:]

if __name__ == '__main__':

    json_output = []

    id = 0
    for ne in networks:
        for sa in samplings:
            for sh in datashares:
                for da in datatypes:
                    for ar in architectures:
                        for lw in loss_weights:
                            for op in optimizers:
                                for lr in lrs:
                                    for nn in nneighbors:
                                        experiment = {
                                                      'id': id,
                                                      'ne': ne,
                                                      'sa': sa,
                                                      'sh': sh,
                                                      'da': da,
                                                      'ar': ar,
                                                      'lw': lw,
                                                      'op': op,
                                                      'lr': lr,
                                                      'nn': nn
                                                     }
                                        json_output.append(experiment)
                                        id = id + 1

    with open('{}.json'.format(experiment_name), 'w') as json_file:
        json_file.write(json.dumps(json_output, indent = 4, sort_keys=True))

    #with open('config.json') as f:
    #  data = json.load(f)
    #print(data[0]['age'])

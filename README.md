# KU-GNN

Download data here: https://www.dropbox.com/s/f4cz75w14pbnniu/data.zip?dl=0

Put in /KU-GNN/code/dgl_dynamic_graph_cnn/data

## TODO:

[done] Create a nearest neighbors graph from a point cloud and visualize https://github.com/epfl-lts2/pygsp/blob/master/pygsp/graphs/nngraphs/nngraph.py

[done] Convert/create dgl style graphs by using xnetwork format https://pygsp.readthedocs.io/en/latest/reference/graphs.html or use dgl.nn.pytorch.KNNGraph to create the graph

[done] Create pytorch dataloader https://towardsdatascience.com/how-to-build-custom-dataloader-for-your-own-dataset-ae7fd9a40be6

[almost] Make implementations more similar/comparable, lacks some data augmentation used for Keras PointNet

[done but code not added yet] Visualize and compare results

[] Why is performance not better?

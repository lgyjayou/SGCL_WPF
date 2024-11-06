# Spatiotemporal Graph Contrastive Learning for Wind Power Forecasting

This is the pytorch implementation of the paper: Spatiotemporal Graph Contrastive Learning for Wind Power Forecasting

## Data Preparation

The dataset SDWPF can be downloaded from [here](https://aistudio.baidu.com/competition/detail/152/0/introduction), and should be put into the `./data/SDWPF` folder. For more information about the SDWPF dataset, please refer to this [link](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2950) or this [paper](https://arxiv.org/abs/2208.04360). 

## Run
Before running, users need to generate two types of static graphs by using the `get_geo_adj.py` file and the `get_dtw_graph.py` file respectively, and then you can run the code with the following command: 
```
python train.py
```
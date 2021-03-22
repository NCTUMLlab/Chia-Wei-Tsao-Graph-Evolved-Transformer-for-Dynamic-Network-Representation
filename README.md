# Chia-Wei-Tsao-Graph-Evolved-Transformer-for-Dynamic-Network-Representation
# Introduction 
This is the source code for the Master thesis Graph-Evolved Transformer for Dynamic Network Representation from National Chiao Tung University, Taiwan.
# Graph-Evolved Transformer for Dynamic Network Representation
This study presents a novel graph representation which tightly integrates the information sources of node embedding and weight matrix in graph learning representation. A new way to updating parameters is proposed by using the transformer decoder rather than using LSTM or GRU. In particular, the positional embedding and masked attention are not required in this setting. The graph structural data are therefore merged in transformer. In addition, the adjacency matrix in each graph structural data is obtained to represent the position information. In learning procedure, the input sample at each time step is only formed by a graph snapshot. Mask module is therefore not required. We build two attention layers, one is to calculate the weight matrix in GNN and the other is to estimate node embedding and weight matrix. The first attention layer is to perform self-attention for weight matrix in GNN. The second attention layer is to find cross attention where both weight matrix and node embedding in GNN are considered. Model robustness can be improved according to these attention layers. Experiments on financial prediction show the merit of the proposed method for temporal information representation based on the evolutionary graph embedded transformer.

![image](https://github.com/NCTU-MLLab/Chia-Wei-Tsao-Graph-Evolved-Transformer-for-Dynamic-Network-Representation/blob/main/1layer.png)

# Getting Start
## Environment
The developed environment is listed in below 

OS : Ubuntu 16.04 

CUDA : 11.1

Nvidia Driver : 455.23

Python 3.6.9

Pytorch 1.2.0

## Setup

Please create the environment with the instruction. With the instruction shown as below you can create the image.

```sh
sudo docker build -t gcn_env:latest docker-set-up/
```

Start the container

```sh
sudo docker run -ti  --gpus all -v $(pwd):/evolveGCN  gcn_env:latest
```

## Training and Evaluation
Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
cd Graph-Evolved Transformer
python run_exp.py --config_file ./experiments/parameters_example.yaml
```

Most of the parameters in the yaml configuration file are self-explanatory. For hyperparameters tuning, it is possible to set a certain parameter to 'None' and then set a min and max value. Then, each run will pick a random value within the boundaries (for example: 'learning_rate', 'learning_rate_min' and 'learning_rate_max').

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. Then, we will have a log file with all information including:

Parameters 

Training result, for example, F1 score,ect.

Validaation result

Testing result

# Reference
[1] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191). AAAI 2020.


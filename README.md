## FedDAD
This project is the code and the supplementary of "Federated Recommendation with Double Additive Decoupling"

## Requirements

1. The code is implemented with `Python >= 3.8` and `torch~=1.13.1+cu117`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

1. First create two folders: `./logs` and `./results`;

2. Put datasets into the path `[parent_folder]/datasets/`;

3. ``````
python PyProject/FedDAD/train.py --alias FedDAD --dataset filmtrust --data_file "PyProject/FedDAD/datasets/filmtrust/ratings.dat" --mu 2e-1 --lambda 2e-1 --l2_regularization 1e-3 --lr_u_p 4e0 --lr_u_c 4e0 --lr_i_p  4e1 --lr_i_c 4e1
   ``````

# FedDAD



> This project is the code and the supplementary of "**Federated Recommendation with Additive Personalization**"

> Notice that <u>FedRAP is highly sensitive to the <font color='red'>**Parameter Combinations**</font>, which may result in significant differences in performance!</u>


![Poster of FedRAP @ ICLR 2024](https://iclr.cc/media/PosterPDFs/ICLR%202024/17446.png)


## Requirements

1. The code is implemented with `Python >= 3.8` and `torch~=1.13.1+cu117`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

1. First create two folders: `./logs` and `./results`;

2. Put datasets into the path `[parent_folder]/datasets/`;

3. ``````
python PyProject/FedDAD/train.py --alias FedDAD --dataset filmtrust --data_file "PyProject/FedDAD/data/filmtrust/ratings.dat" --mu 2e-1 --lambda 2e-1 --l2_regularization 1e-3 --lr_u_p 4e0 --lr_u_c 4e0 --lr_i_p  4e1 --lr_i_c 4e1
   ``````

## Citation

If you find this paper useful in your research, please consider citing:

```
@inproceedings{
    li2024federated,
    title={Federated Recommendation with Additive Personalization},
    author={Zhiwei Li and Guodong Long and Tianyi Zhou},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=xkXdE81mOK}
}
```

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([![Static Badge](https://img.shields.io/badge/Email-lizhw.cs%40outlook.com-red?style=plastic&logo=mail&labelColor=%23FCFAF2&color=grey)](mailto:lizhw.cs@outlook.com))
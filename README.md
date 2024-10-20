# Gradient Rewiring for Editable Graph Neural Network Training (NeurIPS 24)
This repository contains the code and instructions to reproduce the experiments for our NeurIPS 2024 paper [[PDF]](https://openreview.net/pdf?id=XY2qrq7cXM), "Gradient Rewiring for Editable Graph Neural Network Training."

## Dependency
Make sure you have the following dependencies installed:
```
numpy >= 1.24.4
torch >= 2.0.0
pandas >= 2.0.3
scipy >= 1.10.1
ogb >= 1.3.6
``` 

## Reproducing Experimental Results
Follow these steps to reproduce the results presented in the paper:
1. Install the package: At the project root directory, run the following command to install the package:
* Run ```pip install -e .``` at the proj root directory
2. Train the Model: To train a model under the inductive setting, use the following command:
```
bash scripts/run_pretrain.sh $GPU_NUMBER
```

Here, $GPU_NUMBER is the index of the GPU you want to use. For example, running ```bash scripts/run_pretrain.sh 3``` means using GPU 3 to train the models.
3. Perform Model Editing: After training the model, you can perform editing using the following commands:
* Standard evaluation:
```
bash scripts/eval.sh $GPU_NUMBER
```
* Evaluation using Gradient Rewiring (GRE):
```
bash scripts/eval_gre.sh $GPU_NUMBER
```
* Evaluation using the extended version, GRE+:
```
bash scripts/eval_gre_plus.sh $GPU_NUMBER
```


## Citation

```
@inproceedings{
jiang2024gradient,
title={Gradient Rewiring for Editable Graph Neural Network Training},
author={Jiang, Zhimeng and Liu, Zirui and Han, Xiaotian and Feng, Qizhang and Jin, Hongye and Tan, Qiaoyu and Zhou, Kaixiong and Zou, Na and Hu, Xia},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=XY2qrq7cXM}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
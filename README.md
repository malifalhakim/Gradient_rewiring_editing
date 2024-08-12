## Gradient Rewiring for Editable Graph Neural Network Training

# How to reproduce the experimental results

* Run ```pip install -e .``` at the proj root directory

* Then, train a model under inductive setting. 
```
bash scripts/run_pretrain.sh $GPU_NUMBER
```

GPU_NUMBER is the index of GPU to use. For example, ```bash scripts/run_pretrain.sh 3``` means using GPU 3 to train the models.

* Then, perform the editing on pretrained model.

```
bash scripts/eval.sh $GPU_NUMBER
bash scripts/eval_gre.sh $GPU_NUMBER
bash scripts/eval_gre_plus.sh $GPU_NUMBER
```
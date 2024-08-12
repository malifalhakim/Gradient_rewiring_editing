import argparse
import pdb
import yaml
import torch
from editable_gnn import smat_util
import numpy as np
import editable_gnn.models as models
from data import get_data, prepare_dataset
from editable_gnn import WholeGraphTrainer, BaseTrainer, set_seeds_all


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='/data/username/dataset/graphdata')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--runs', default=1, type=int,
                    help='number of runs')
parser.add_argument('--hyper_Diff', default=1.0, type=float, help='the hyperparameter for Diff loss')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='hyperparameter for kl distance power')
parser.add_argument('--output_dir', default='./ckpts', type=str)

parser.add_argument('--giant', action='store_true', default=False, 
                        help='Giant indicator with default False')
parser.add_argument('--node_emb_path', default='/data/username/dataset/graphdata/GIANT/ogbn-arxiv/X.all.xrt-emb.npy', type=str)



if __name__ == '__main__':
    args = parser.parse_args()
    set_seeds_all(args.seed)
    with open(args.config, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
    print(args)
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    MODEL_FAMILY = getattr(models, model_config['arch_name'])
    data, num_features, num_classes = get_data(args.root, args.dataset)
    
    if args.giant:
        data.x = torch.from_numpy(smat_util.load_matrix(args.node_emb_path).astype(np.float32))
        print("Loaded pre-trained node embeddings of shape={} from {}".format(data.x.shape, args.node_emb_path))
        
    model = MODEL_FAMILY(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    model.cuda()
    print(model)
    train_data, whole_data = prepare_dataset(model_config, data, remove_gree_index=True)
    del data
    print(f'training data: {train_data}')
    print(f'whole data: {whole_data}')
    TRAINER_CLS = BaseTrainer if  model_config['arch_name'] == 'MLP' else WholeGraphTrainer
    trainer = TRAINER_CLS(args, model, train_data, whole_data, model_config, 
                          args.output_dir, args.dataset, multi_label, 
                          False)

    trainer.train()
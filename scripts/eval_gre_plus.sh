save_dir_root=./ckpts
output_dir=./finetune
criterion=wrong2correct

## cora flickr reddit2 arxiv amazoncomputers amazonphoto coauthorcs coauthorphysics products  yelp

manner=GRE_Plus
for train_split in 2 3 5; do ### 2 3 5
for gamma in 1.0; do ##0. 0.1 1.0 2.0 5.0 10.0 50.0
for dataset in cora; do ### cora flickr reddit2 arxiv
for model in gcn sage; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}/${manner}" ]; then
        mkdir -p "./${output_dir}/${dataset}/${manner}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --gamma ${gamma} \
        --train_split ${train_split} \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval_train_split=${train_split}_gamma=${gamma}.log
done
wait
done
done
done
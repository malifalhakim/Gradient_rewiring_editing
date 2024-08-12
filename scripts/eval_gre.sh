save_dir_root=./ckpts
output_dir=./finetune
criterion=wrong2correct


# cora flickr arxiv amazoncomputers amazonphoto coauthorcs coauthorphysics reddit2 products  yelp

manner=GRE
for gamma in 0. 0.1 1.0 10.0 50.0; do
for dataset in cora; do ### cora flickr reddit2 arxiv
for model in gcn sage; do ###gcn sage mlp
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
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval_gamma=${gamma}.log
done
wait
done
done

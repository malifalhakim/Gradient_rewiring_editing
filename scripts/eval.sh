save_dir_root=./ckpts
output_dir=./finetune

criterion=wrong2correct

## cora flickr arxiv amazoncomputers amazonphoto coauthorcs coauthorphysics reddit2 products 

for manner in GD; do    ### GD
for dataset in cora flickr; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto
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
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval.log
done
wait
done
done


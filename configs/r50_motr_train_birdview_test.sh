# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for MOT17

PRETRAIN=r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=exps/birdview_container
python -m pdb \
     main.py \
    --meta_arch motr \
    --dataset_file birdview \
    --epoch 50 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-5 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 150 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path ./datasets/data_path \
    --data_txt_path_train ./datasets/data_path/birdview.val \
    --data_txt_path_val ./datasets/data_path/birdview.val \
    --num_workers 0 \
    --output_dir ${EXP_DIR}/
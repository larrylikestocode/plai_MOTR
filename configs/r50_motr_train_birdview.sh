# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for MOT17

PRETRAIN=r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
#EXP_DIR=mlexps16/birdview
EXP_DIR=$1
python  \
     main.py \
    --meta_arch motr \
    --dataset_file birdview \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr $2 \
    --lr_backbone $3 \
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
    --mot_path ./datasets/data_path/dist/ \
    --data_txt_path_train ./datasets/data_path/dist/birdview_itra.train \
    --data_txt_path_val ./datasets/data_path/dist/birdview_itra.val \
    --data_txt_path_test ./datasets/data_path/dist/birdview_interpret.test \
    --num_workers 0 \
    --output_dir ${EXP_DIR}/ \
    --birdview_type itra \
    --save_period 5 \
    --use_wandb 1 \

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 1000 --lr_TB 0.002 --lr_IB 0.002 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot -1 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 0 &
CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 1000 --lr_TB 0.002 --lr_IB 0.002 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot -1 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 0 &
CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 1000 --lr_TB 0.02  --lr_IB 0.02  --comment 26head --pooling_strategy learnablePrompt_multi --num_shot -1 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 0 &

CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 0 &
CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 1 &
CUDA_VISIBLE_DEVICES=0 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 2 &
CUDA_VISIBLE_DEVICES=1 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 3 &
CUDA_VISIBLE_DEVICES=1 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.2 --p_bag_drop_out 0.2 --weight_lossA 25 --seed 4 &

CUDA_VISIBLE_DEVICES=1 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 0 &
CUDA_VISIBLE_DEVICES=2 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 1 &
CUDA_VISIBLE_DEVICES=2 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 2 &
CUDA_VISIBLE_DEVICES=3 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 3 &
CUDA_VISIBLE_DEVICES=3 python -u train_TCGAFeat_MIL_CLIP.py --epochs 8000 --lr_TB 0.02 --lr_IB 0.02 --comment 26head --pooling_strategy learnablePrompt_multi --num_shot 16 --p_drop_out 0.0 --p_bag_drop_out 0.0 --weight_lossA 25 --seed 4 &

wait

torchrun --nproc_per_node=4 train.py\
    --data-path /home/cs/Documents/datasets/imagenet\
    --model mobilenet_v2  --output-dir mobilenet_v2 --weights MobileNet_V2_Weights.IMAGENET1K_V1\
    --batch-size 96 --epochs 10 --opt adamw --lr-warmup-epochs 1 --lr-warmup-decay 0.0 --lr 1e-8 --lr-step-size 10 --lr-gamma 0.5 --wd 0.00004


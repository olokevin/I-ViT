### Training
choices=['deit_tiny', 'deit_small', 'deit_base', 'swin_tiny', 'swin_small', 'swin_base', 'vit_base', 'vit_large'],
CUDA_VISIBLE_DEVICES=2 python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 30 --lr 1e-6 --batch-size 64 
CUDA_VISIBLE_DEVICES=2 nohup python quant_train.py --model deit_small --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 30 --lr 1e-6 --batch-size 64 >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python quant_train.py --model deit_base --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 30 --lr 1e-6 --batch-size 32 >/dev/null 2>&1 &

### Test
CUDA_VISIBLE_DEVICES=2 python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 1 --lr 1e-6 --batch-size 64 --resume results/deit_tiny_checkpoint.pth.tar --no-train

### imagenet-c
# FP32
CUDA_VISIBLE_DEVICES=2 python quant_train.py --model fp32_deit_tiny_patch16_224 --data /home/yequan_zhao/dataset --data-set imagenet_c-3-5000-gaussian_noise --epochs 30 --lr 1e-3 --batch-size 100 --opt sgd
CUDA_VISIBLE_DEVICES=2 nohup python quant_train.py --model fp32_deit_tiny_patch16_224 --data /home/yequan_zhao/dataset --data-set imagenet_c-3-5000-gaussian_noise --epochs 30 --lr 1e-5 --batch-size 20 >/dev/null 2>&1 &
# INT8
CUDA_VISIBLE_DEVICES=2 python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset --data-set imagenet_c-3-5000-gaussian_noise --resume results/deit_tiny_checkpoint.pth.tar --epochs 30 --batch-size 100 --lr 1e-2 --opt sgd --real-quant
CUDA_VISIBLE_DEVICES=2 nohup python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset --data-set imagenet_c-3-5000-gaussian_noise --resume results/deit_tiny_checkpoint.pth.tar --epochs 30 --batch-size 100 --lr 1e-2 --opt sgd --real-quant >/dev/null 2>&1 &
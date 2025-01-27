CUDA_VISIBLE_DEVICES=2 python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 30 --lr 1e-6 --batch-size 64 

CUDA_VISIBLE_DEVICES=2 python quant_train.py --model deit_tiny --data /home/yequan_zhao/dataset/ImageNet2012 --epochs 1 --lr 1e-6 --batch-size 64 --resume results/deit_tiny_checkpoint.pth.tar --no-train
import argparse
import os
import time
import math
import logging
import numpy as np

import torch
import torch.nn as nn
from pathlib import Path

import timm
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma, accuracy
# from timm.utils import NativeScaler

from models import *
from utils import *

import random
import easydict
from quant.real_quant import replace_Quant_with_RealQuant, RealQuant_Scaler, NativeScaler
from quant.real_quant import efficient_real_quant_perturb_parameters, efficient_real_quant_gen_grad

# DEBUG = False
DEBUG = True


parser = argparse.ArgumentParser(description="I-ViT")

parser.add_argument("--model", default='deit_tiny',
                    # choices=['deit_tiny', 'deit_small', 'deit_base', 
                    #          'swin_tiny', 'swin_small', 'swin_base',
                    #          'vit_base', 'vit_large'],
                    help="model")
parser.add_argument('--data', metavar='DIR', default='/dataset/imagenet/',
                    help='path to dataset')
# parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET'],
#                     type=str, help='Image Net dataset path')
parser.add_argument('--data-set', default='IMNET',
                    type=str, help='Image Net dataset path')
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=1000,
                    type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--output-dir', type=str, default='results/',
                    help='path to save log and quantized model')

parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
# parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
#                     help='Drop path rate (default: 0.1)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.)')


parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=5e-7, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                           "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument('--best-acc1', type=float, default=0, help='best_acc1')

parser.add_argument('--no-train', action='store_true')

parser.add_argument('--real-quant', action='store_true')

parser.add_argument('--en-ZO', action='store_true')


def str2model(name):
    d = {'deit_tiny': deit_tiny_patch16_224,
         'deit_small': deit_small_patch16_224,
         'deit_base': deit_base_patch16_224,
         'swin_tiny': swin_tiny_patch4_window7_224,
         'swin_small': swin_small_patch4_window7_224,
         'swin_base': swin_base_patch4_window7_224,
         'vit_base': vit_base_patch16_224,
         'vit_large': vit_large_patch16_224,
         }
    print('Model: %s' % d[name].__name__)
    return d[name]

# def set_torch_deterministic(random_state: int = 0) -> None:
#     random_state = int(random_state) % (2 ** 32)
#     # random_state = int(random_state)
#     torch.manual_seed(random_state)
#     np.random.seed(random_state)
#     if torch.cuda.is_available():
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.cuda.manual_seed_all(random_state)
#     random.seed(random_state)

def main():
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings('ignore')
    
    args.output_dir = os.path.join(args.output_dir, args.model, str(os.getpid()))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(args.output_dir + '/log.log'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info(str(os.getpid()))

    device = torch.device(args.device)

    # Dataset
    train_loader, val_loader = dataloader(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Model
    if 'fp32' in args.model:
        model_name = args.model.split('_', 1)[-1]
        model = timm.create_model(model_name, pretrained=True)
    else:
        model = str2model(args.model)(pretrained=True,
                                      num_classes=args.nb_classes,
                                      drop_rate=args.drop,
                                      drop_path_rate=args.drop_path)
    
    model = model.to(device)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=device)
        
        for key in checkpoint.keys():
            if 'act_scaling_factor' in key:
                checkpoint[key] = torch.tensor([checkpoint[key]], device=device)
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint, strict=False, assign=True)
    
        ### RealQuant
        if args.real_quant:
            replace_Quant_with_RealQuant(model)
            
    model = model.to(device)
    
    
    
    
    ### trainable parameters
    param_dim = 0
    for name, param in model.named_parameters():
        if 'cls_token' in name:
            # param.requires_grad = True
            param.requires_grad = False
        # elif 'patch_embed' in name:
        #     param.requires_grad = True
        
        # elif 'blocks.0' in name:
        #     param.requires_grad = True
        
        # elif 'blocks.11' in name and 'weight' in name:
        elif 'blocks.11.attn.proj.weight' in name:
            param.requires_grad = True
            param_dim += int(param.numel())
        
        else:
            param.requires_grad = False
            # param.requires_grad = True
    
    ### ZO training
    if args.en_ZO:
        # ZO_config = {
        #     'n_sample': 1000,
        #     'sigma': 1,
        #     'param_dim': param_dim,
        # }
        ZO_config = {
          "name": "ZO_Estim_MC",
          "sigma": 1,
          "n_sample": 10,
          "signsgd": False,
          "scale": 'dim',
          "ZO_trainable_layers_list": ["RealQuantLinear"],
          
          # "actv_perturb_block_idx_list": None,
          # "param_perturb_block_idx_list": "all",
          
          "actv_perturb_block_idx_list": "all",
          "param_perturb_block_idx_list": None,
          
          "obj_fn_type": "classifier",
          # "estimate_method": "forward",
          "estimate_method": "antithetic",
          "sample_method": "bernoulli",
          "en_layerwise_perturbation": True,
          "en_partial_forward": False,
          "quantized": False,
          "normalize_perturbation": False,
          "en_param_commit": False
      }
        ZO_config = easydict.EasyDict(ZO_config)
        
    else:
        ZO_config = None
    
    # ================== ZO_Estim ======================
    
    if ZO_config is not None:
        from ZO_Estim.ZO_Estim_entry import build_ZO_Estim
        ZO_Estim = build_ZO_Estim(ZO_config, model=model, )
    else:
        ZO_Estim = None
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        
    args.min_lr = args.lr / 15
    optimizer = create_optimizer(args, model)
    if args.real_quant:
        # loss_scaler = RealQuant_NativeScaler(model)
        loss_scaler = RealQuant_Scaler(model)
    else:
        # loss_scaler = NativeScaler()
        loss_scaler = NativeScaler(model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.en_ZO:
        criterion = nn.CrossEntropyLoss()
    else:
        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
            
    criterion_v = nn.CrossEntropyLoss()
    
    ### old resume training
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1
    #         if args.model_ema:
    #             load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
    #         if 'scaler' in checkpoint:
    #             loss_scaler.load_state_dict(checkpoint['scaler'])
    #     lr_scheduler.step(args.start_epoch)
    
    # acc1 = validate(args, val_loader, model, criterion_v, device)
    # logging.info(f'Initial Acc at epoch 0: {acc1}')

    print(f"Start training for {args.epochs} epochs")
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.no_train:
            pass
        else:
            train(args, train_loader, model, criterion, optimizer, epoch,
                  loss_scaler, args.clip_grad, model_ema, mixup_fn, device, ZO_config, ZO_Estim)
            lr_scheduler.step(epoch)

        # if args.output_dir:  # this is for resume training
        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'model_ema': get_state_dict(model_ema),
        #         'scaler': loss_scaler.state_dict(),
        #         'args': args,
        #     }, checkpoint_path)

        acc1 = validate(args, val_loader, model, criterion_v, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > args.best_acc1
        args.best_acc1 = max(acc1, args.best_acc1)
        if is_best:
            # record the best epoch
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model}_checkpoint.pth.tar'))
        logging.info(f'Acc at epoch {epoch+1}: {acc1}')
        logging.info(f'Best acc at epoch {best_epoch+1}: {args.best_acc1}')


def train(args, train_loader, model, criterion, optimizer, epoch, loss_scaler, max_norm, model_ema, mixup_fn, device, ZO_config=None, ZO_Estim=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if 'imagenet_c' in args.data_set:
        freeze_model(model)
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    else:
        unfreeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if mixup_fn is not None:
            data, target = mixup_fn(data, target)
            
        if ZO_Estim is not None:
            from ZO_Estim.ZO_Estim_entry import build_obj_fn
            from ZO_Estim.ZO_utils import default_create_bwd_pre_hook_ZO_grad
            obj_fn = build_obj_fn(ZO_Estim.obj_fn_type, data=data, target=target, model=model, criterion=criterion)
            ZO_Estim.update_obj_fn(obj_fn)
            with torch.no_grad():
                pred, loss = obj_fn()
                ZO_Estim.estimate_grad(old_loss=loss)
                
            ### pseudo NP
            if ZO_Estim.splited_layer_list is not None:
                # bwd_pre_hook_list = []
                # for splited_layer in ZO_Estim.splited_layer_list:
                #     create_bwd_pre_hook_ZO_grad = getattr(splited_layer.layer, 'create_bwd_pre_hook_ZO_grad', default_create_bwd_pre_hook_ZO_grad)
                #     bwd_pre_hook_list.append(splited_layer.layer.register_full_backward_pre_hook(create_bwd_pre_hook_ZO_grad(splited_layer.layer.ZO_grad_output, DEBUG)))
                # output = model(data)
                # loss = criterion(output, target)
                # loss.backward()
                
                # for bwd_pre_hook in bwd_pre_hook_list:
                #     bwd_pre_hook.remove()
                
                fwd_hook_list = []
                for splited_layer in ZO_Estim.splited_layer_list:

                    fwd_hook_get_param_grad = splited_layer.layer.create_fwd_hook_get_param_grad(splited_layer.layer.ZO_grad_output, DEBUG)
                    fwd_hook_list.append(splited_layer.layer.register_forward_hook(fwd_hook_get_param_grad))
                    
                    with torch.no_grad():
                        output = model(data)
                        loss = criterion(output, target)
                
                for fwd_hook_handle in fwd_hook_list:
                    fwd_hook_handle.remove()
            
            ### save param FO grad
            if DEBUG:
                for param in model.parameters():
                    if param.requires_grad:
                        param.ZO_grad = param.grad.clone()
                        
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                for param in model.parameters():
                    if param.requires_grad:
                        param.FO_grad = param.grad.clone()
                
                optimizer.zero_grad()
            
            ### print FO ZO grad
                print('param cos sim')
                for param in model.parameters():
                    if param.requires_grad:
                        print(f'{F.cosine_similarity(param.FO_grad.view(-1), param.ZO_grad.view(-1), dim=0)}')
                    
                print('param Norm ZO/FO: ')
                for param in model.parameters():
                    if param.requires_grad:
                        print(f'{torch.linalg.norm(param.ZO_grad.view(-1)) / torch.linalg.norm(param.FO_grad.view(-1))}')
                
                optimizer.zero_grad()
            
        # if args.en_ZO:
            
            # seed_list = []
            # loss_diff_list = []
            # # no scale
            # # grad_scale = 1 / ZO_config.n_sample
            # # sqrt_fim
            # # grad_scale = 1 / (ZO_config.n_sample + ZO_config.param_dim - 1)
            # # dim
            # grad_scale = 1 / math.sqrt( (ZO_config.n_sample) * (ZO_config.n_sample + ZO_config.param_dim - 1) )
            
            # model.eval()
            # with torch.no_grad():
            #     output = model(data)
            #     loss = criterion(output, target)
                
            #     for _ in range(ZO_config.n_sample):
            #         random_seed = np.random.randint(1000000000)
            #         efficient_real_quant_perturb_parameters(model, random_seed, ZO_config.sigma)
            #         output = model(data)
            #         pos_loss = criterion(output, target)
            #         efficient_real_quant_perturb_parameters(model, random_seed, -ZO_config.sigma)
                    
            #         # efficient_real_quant_perturb_parameters(model, random_seed, -sigma)
            #         # output = model(data)
            #         # neg_loss = criterion(output, target)
            #         # efficient_real_quant_perturb_parameters(model, random_seed, sigma)
                    
            #         seed_list.append(random_seed)
            #         loss_diff_list.append((pos_loss - loss) * grad_scale)
                
            #     efficient_real_quant_gen_grad(model, loss_diff_list, seed_list, lr=None)
            
            # # model.train()
            
            ### test
            # ZO_grad = model.blocks[0].attn.qkv.weight.grad.clone().view(-1)
            # optimizer.zero_grad()
            
            # output = model(data)
            # loss = criterion(output, target)

            # # compute gradient and do SGD step
            # FO_loss_scaler = NativeScaler(model)
            # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            # FO_loss_scaler(loss, optimizer, clip_grad=max_norm,
            #             parameters=model.parameters(), create_graph=is_second_order)

            # FO_grad = model.blocks[0].attn.qkv.weight.grad.clone().view(-1)
            
            # print(f'ZO_norm / FO_norm: {torch.norm(ZO_grad) / torch.norm(FO_grad)}')
            # # print cosine similarity between ZO_grad and FO_grad
            # cosine_similarity = torch.nn.functional.cosine_similarity(ZO_grad, FO_grad, dim=0)
            # print(f'Cosine similarity between ZO_grad and FO_grad: {cosine_similarity.item()}')
            # print('test')
            
        else:
            output = model(data)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)
        
        # torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model}_checkpoint.pth.tar'))


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    freeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == "__main__":
    main()

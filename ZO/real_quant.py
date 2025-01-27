import decimal

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from transformers.models.ibert.quant_modules import QuantEmbedding
from models.quantization_utils.quant_modules import QuantLinear
from models.quantization_utils.quant_utils import SymmetricQuantFunction, symmetric_linear_quantization_params

class RealQuantEmbedding(nn.Module):
    """
    Quantized version of `torch.nn.Embedding`. Adds quantization-specific arguments on top of `torch.nn.Embedding`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        fp_weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        weight_bit=8,
        momentum=0.95,
        quant_mode=False,
    ):
        super().__init__()
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        w = fp_weight
        w_transform = w.data.detach()
        w_min = w_transform.min().expand(1)
        w_max = w_transform.max().expand(1)

        self.scale_w = symmetric_linear_quantization_params(weight_bit, w_min, w_max)
        weight_integer = SymmetricQuantFunction.apply(
           fp_weight, weight_bit, self.scale_w, True
        )
        self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
        self.weight.data.copy_(weight_integer)

        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.percentile_mode = False

    def forward(self, x, positions=None, incremental_state=None):
        if not self.quant_mode:
            return (
                nn.functional.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ),
                None,
            )

        emb_int = nn.functional.embedding(
            x,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return emb_int * self.scale_w, self.scale_w

    def wp_add_perturbation(self, sigma, rand_gen_fn, seed=None):
        # if seed is not None:
        #     state = torch.get_rng_state()
        #     torch.manual_seed(seed)
        
        z = rand_gen_fn(self.weight.shape)
        self.weight.data += sigma * z
        self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
        
        # if seed is not None:
        #     torch.set_rng_state(state)
    
    def wp_gen_grad(self, loss_diff, rand_gen_fn, lr):
            
        if lr is not None:
            z = rand_gen_fn(self.weight.shape)
            self.weight.data -= lr * loss_diff / (self.scale_w ** 2) * z
            self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
            
        else:
            if self.weight.grad is None:
                self.weight.grad = loss_diff / (self.scale_w ** 2) * rand_gen_fn(self.weight.shape)
            else:
                self.weight.grad += loss_diff / (self.scale_w ** 2) * rand_gen_fn(self.weight.shape)

class RealQuantLinear(nn.Module):
    """
    Quantized version of `torch.nn.Linear`. Adds quantization-specific arguments on top of `torch.nn.Linear`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        bias_bit (`int`, *optional*, defaults to `32`):
            Bitwidth for the quantized bias.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether or not to use channel-wise quantization.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(
        self, in_features, out_features, fp_weight, fp_bias=None, weight_bit=8, bias_bit=32, per_channel=False, quant_mode=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        w = fp_weight
        w_transform = w.data.detach()
        if per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
            
        self.scale_w = symmetric_linear_quantization_params(weight_bit, w_min, w_max)
        self.weight = nn.Parameter(SymmetricQuantFunction.apply(
              fp_weight, weight_bit, self.scale_w, True
            )
        )

        if fp_bias is not None:
            self.bias = nn.Parameter(fp_bias)
            self.register_buffer("bias_integer", torch.zeros_like(self.bias))

        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quant_mode = quant_mode
        self.percentile_mode = False
        self.weight_function = SymmetricQuantFunction.apply

    def __repr__(self):
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, quant_mode={self.quant_mode})"
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if not self.quant_mode:
            return nn.functional.linear(x, weight=self.weight, bias=self.bias), None

        # assert that prev_act_scaling_factor is a scalar tensor
        assert prev_act_scaling_factor is not None and prev_act_scaling_factor.shape == (1,), (
            "Input activation to the QuantLinear layer should be globally (non-channel-wise) quantized. "
            "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        )

        bias_scaling_factor = self.scale_w * prev_act_scaling_factor
        self.bias_scaling_factor = bias_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor, True)

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return (
            nn.functional.linear(x_int, weight=self.weight, bias=self.bias_integer) * bias_scaling_factor,
            bias_scaling_factor,
        )
    
    def wp_add_perturbation(self, sigma, rand_gen_fn, seed=None):
        # if seed is not None:
        #     state = torch.get_rng_state()
        #     torch.manual_seed(seed)
        
        z = rand_gen_fn(self.weight.shape)
        self.weight.data += sigma * z
        self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
        if self.bias is not None:
            self.bias.data += sigma * self.bias_scaling_factor * rand_gen_fn(self.bias.shape)
        
        # if seed is not None:
        #     torch.set_rng_state(state)
    
    def wp_gen_grad(self, loss_diff, rand_gen_fn, lr):
            
        if lr is not None:
            z = rand_gen_fn(self.weight.shape)
            self.weight.data -= lr * loss_diff / (self.scale_w.view(-1,1) ** 2) * z
            self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
            
            if self.bias is not None:
                self.bias.data -= lr * loss_diff / self.bias_scaling_factor * rand_gen_fn(self.bias.shape)
                # self.bias.data = self.bias.data.round().clamp(- 2 ** (self.bias_bit - 1), 2 ** (self.bias_bit - 1) - 1)
        else:
            if self.weight.grad is None:
                self.weight.grad = loss_diff / (self.scale_w.view(-1,1) ** 2) * rand_gen_fn(self.weight.shape)
            else:
                self.weight.grad += loss_diff / (self.scale_w.view(-1,1) ** 2) * rand_gen_fn(self.weight.shape)
                
            if self.bias is not None:
                if self.bias.grad is None:
                    self.bias.grad = loss_diff / self.bias_scaling_factor * rand_gen_fn(self.bias.shape)
                else:
                    self.bias.grad += loss_diff / self.bias_scaling_factor * rand_gen_fn(self.bias.shape)

def default_wp_add_perturbation(module, sigma, rand_gen_fn, seed=None):
    # if seed is not None:
    #     state = torch.get_rng_state()
    #     torch.manual_seed(seed)
    for param in module.parameters():
        if param.requires_grad:
            perturbation = rand_gen_fn(param.shape)
            param.data += sigma * perturbation
    
    # if seed is not None:
    #     torch.set_rng_state(state)
            
def default_wp_gen_grad(module, loss_diff, rand_gen_fn, lr=None): 
    if lr is not None:
        for param in module.parameters():
            if param.requires_grad:
                perturbation = rand_gen_fn(param.shape)
                param.data -= lr * loss_diff * perturbation
    else:
        for param in module.parameters():
            if param.requires_grad:
                perturbation = rand_gen_fn(param.shape)
                if param.grad is None:
                    param.grad = loss_diff * perturbation
                else:
                    param.grad += loss_diff * perturbation

def efficient_real_quant_perturb_parameters(model: nn.Module, random_seed: int, sigma: float):
    torch.manual_seed(random_seed)
    rand_gen_fn = build_rand_gen_fn(sample_method='bernoulli', device='cuda')
    fp_scale = 1e-3
    # fp_scale = 1
    
    for name, module in model.named_modules():
        if isinstance(module, (RealQuantEmbedding, RealQuantLinear)):
            module.wp_add_perturbation(sigma=sigma, rand_gen_fn=rand_gen_fn)
        # elif isinstance(module, (nn.LayerNorm, nn.Linear)):
        elif isinstance(module, (nn.Linear)):
            default_wp_add_perturbation(module, sigma*fp_scale, rand_gen_fn)

    return model

def efficient_real_quant_gen_grad(model: nn.Module, loss_diff_list, seed_list, lr=None):
    fp_scale = 1e-3
    # fp_scale = 1
    
    for i in range(len(loss_diff_list)):
        loss_diff = loss_diff_list[i]
        torch.manual_seed(seed_list[i])
        rand_gen_fn = build_rand_gen_fn(sample_method='bernoulli', device='cuda')
        
        for name, module in model.named_modules():
            if isinstance(module, (RealQuantEmbedding, RealQuantLinear)):
                module.wp_gen_grad(loss_diff, rand_gen_fn, lr)
            # elif isinstance(module, (nn.LayerNorm, nn.Linear)):
            elif isinstance(module, (nn.Linear)):
                default_wp_gen_grad(module, loss_diff, rand_gen_fn, lr / fp_scale)
        
        

def replace_Quant_with_RealQuant(net):
    for m in net.modules():
        to_update_dict = {}
        to_update_embedding_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, QuantLinear):
                to_update_dict[name] = sub_module
            elif isinstance(sub_module, QuantEmbedding):
                to_update_embedding_dict[name] = sub_module
                
        for name, sub_module in to_update_dict.items():
            # if sub_module.bias is not None:
            #     bias = sub_module.bias
            # else:
            #     bias = None
            m._modules[name] = RealQuantLinear(
                sub_module.in_features,
                sub_module.out_features,
                sub_module.weight,
                sub_module.bias,
                sub_module.weight_bit,
                sub_module.bias_bit,
                sub_module.per_channel,
                sub_module.quant_mode,
            )
            # load requires_grad
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
        
        for name, sub_module in to_update_embedding_dict.items():
            m._modules[name] = RealQuantEmbedding(
                sub_module.num_,
                sub_module.dim,
                sub_module.weight,
                sub_module.padding_idx,
                sub_module.max_norm,
                sub_module.norm_type,
                sub_module.scale_grad_by_freq,
                sub_module.sparse,
                None,
                sub_module.weight_bit,
                sub_module.momentum,
                sub_module.quant_mode,
            )
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad

def build_rand_gen_fn(sample_method, device, sampler=None):
    def _rand_gen_fn(shape):
        if sample_method == 'bernoulli':
            ### Rademacher
            sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
        else:
            return NotImplementedError('Unlnown sample method', sample_method)
        
        return sample
    return _rand_gen_fn

# def build_rand_gen_fn(sample_method, device, sampler=None):
#     def _rand_gen_fn(shape):
#         if sample_method == 'bernoulli':
#             ### Rademacher
#             sample = torch.ones(shape) - 2*torch.bernoulli(0.5*torch.ones(shape))
#         else:
#             return NotImplementedError('Unlnown sample method', sample_method)
        
#         return sample.to(device)
#     return _rand_gen_fn
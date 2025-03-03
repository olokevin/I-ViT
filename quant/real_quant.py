import decimal

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from transformers.models.ibert.quant_modules import QuantEmbedding
from models.quantization_utils.quant_modules import QuantLinear
from models.quantization_utils.quant_utils import SymmetricQuantFunction, symmetric_linear_quantization_params

# class RealQuantEmbedding(nn.Module):
#     """
#     Quantized version of `torch.nn.Embedding`. Adds quantization-specific arguments on top of `torch.nn.Embedding`.

#     Args:
#         weight_bit (`int`, *optional*, defaults to `8`):
#             Bitwidth for the quantized weight.
#         momentum (`float`, *optional*, defaults to `0.95`):
#             Momentum for updating the activation quantization range.
#         quant_mode (`bool`, *optional*, defaults to `False`):
#             Whether or not the layer is quantized.
#     """

#     def __init__(
#         self,
#         num_embeddings,
#         embedding_dim,
#         fp_weight,
#         padding_idx=None,
#         max_norm=None,
#         norm_type=2.0,
#         scale_grad_by_freq=False,
#         sparse=False,
#         _weight=None,
#         weight_bit=8,
#         momentum=0.95,
#         quant_mode=False,
#     ):
#         super().__init__()
#         self.num_ = num_embeddings
#         self.dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.max_norm = max_norm
#         self.norm_type = norm_type
#         self.scale_grad_by_freq = scale_grad_by_freq
#         self.sparse = sparse
        
#         w = fp_weight
#         w_transform = w.data.detach()
#         w_min = w_transform.min().expand(1)
#         w_max = w_transform.max().expand(1)

#         self.scale_w = symmetric_linear_quantization_params(weight_bit, w_min, w_max)
#         weight_integer = SymmetricQuantFunction.apply(
#            fp_weight, weight_bit, self.scale_w, True
#         )
#         self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
#         self.weight.data.copy_(weight_integer)

#         self.weight_bit = weight_bit
#         self.momentum = momentum
#         self.quant_mode = quant_mode
#         self.percentile_mode = False

#     def forward(self, x, positions=None, incremental_state=None):
#         if not self.quant_mode:
#             return (
#                 nn.functional.embedding(
#                     x,
#                     self.weight,
#                     self.padding_idx,
#                     self.max_norm,
#                     self.norm_type,
#                     self.scale_grad_by_freq,
#                     self.sparse,
#                 ),
#                 None,
#             )

#         emb_int = nn.functional.embedding(
#             x,
#             self.weight,
#             self.padding_idx,
#             self.max_norm,
#             self.norm_type,
#             self.scale_grad_by_freq,
#             self.sparse,
#         )
#         return emb_int * self.scale_w, self.scale_w

#     def wp_add_perturbation(self, sigma, rand_gen_fn, seed=None):
#         # if seed is not None:
#         #     state = torch.get_rng_state()
#         #     torch.manual_seed(seed)
        
#         z = rand_gen_fn(self.weight.shape)
#         self.weight.data += sigma * z
#         self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
        
#         # if seed is not None:
#         #     torch.set_rng_state(state)
    
#     def wp_gen_grad(self, loss_diff, rand_gen_fn, lr):
            
#         if lr is not None:
#             z = rand_gen_fn(self.weight.shape)
#             self.weight.data -= lr * loss_diff / (self.scale_w ** 2) * z
#             self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
            
#         else:
#             if self.weight.grad is None:
#                 self.weight.grad = loss_diff / (self.scale_w ** 2) * rand_gen_fn(self.weight.shape)
#             else:
#                 self.weight.grad += loss_diff / (self.scale_w ** 2) * rand_gen_fn(self.weight.shape)

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
        self.scale_bias = bias_scaling_factor

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
        
        if self.weight.requires_grad:
            z = rand_gen_fn(self.weight.shape)
            self.weight.data += sigma * z
            self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
        
        if self.bias is not None and self.bias.requires_grad:
            self.bias.data += sigma * self.bias_scaling_factor * rand_gen_fn(self.bias.shape)
        
        # if seed is not None:
        #     torch.set_rng_state(state)
    
    # def wp_gen_grad(self, loss_diff, rand_gen_fn, lr):
            
    #     if lr is not None:
    #         z = rand_gen_fn(self.weight.shape)
    #         self.weight.data -= lr * loss_diff / (self.scale_w.view(-1,1) ** 2) * z
    #         self.weight.data = self.weight.data.round().clamp(- 2 ** (self.weight_bit - 1), 2 ** (self.weight_bit - 1) - 1)
            
    #         if self.bias is not None:
    #             self.bias.data -= lr * loss_diff / self.bias_scaling_factor * rand_gen_fn(self.bias.shape)
    #             # self.bias.data = self.bias.data.round().clamp(- 2 ** (self.bias_bit - 1), 2 ** (self.bias_bit - 1) - 1)
    #     else:
    #         ### quantization-aware scaling in pre_step of optimizer
    #         if self.weight.requires_grad:
    #             z = rand_gen_fn(self.weight.shape)
    #             if self.weight.grad is None:
    #                 self.weight.grad = loss_diff * z
    #             else:
    #                 self.weight.grad += loss_diff * z
                
    #         if self.bias is not None and self.bias.requires_grad:
    #             if self.bias.grad is None:
    #                 self.bias.grad = loss_diff * rand_gen_fn(self.bias.shape)
    #             else:
    #                 self.bias.grad += loss_diff * rand_gen_fn(self.bias.shape)
    
    @staticmethod
    def create_fwd_hook_get_out_dimension():
        def fwd_hook(module, input, output):
            # input is a tuple
            module.output_shape = output[0].shape
            module.out_dimension = output[0].numel() / output[0].shape[0]
        return fwd_hook
    
    @staticmethod
    def create_fwd_hook_add_perturbation(seed, sigma, rand_gen_fn):
        def fwd_hook(module, input, output):
            # input is a tuple
            # module.in_value = input[0].detach().clone()
            # output is a tensor
            perturbation_shape = output[0].shape
            
            perturb_output = []
            
            state = torch.get_rng_state()
            torch.manual_seed(seed)
            perturbation = rand_gen_fn(perturbation_shape)
            # perturb_output.append(output[0] + sigma * perturbation)
            perturb_output.append(output[0] + sigma * perturbation * module.scale_bias)
            torch.set_rng_state(state)
            module.perturbation = perturbation
            
            perturb_output.append(output[-1])
            perturb_output=tuple(perturb_output)
            return perturb_output
        return fwd_hook
        
    @staticmethod
    def create_bwd_pre_hook_ZO_grad(ZO_grad_output, debug=False):
        def bwd_pre_hook(module, grad_output):
            if debug:
                print(f'{torch.nn.functional.cosine_similarity(grad_output[0].view(-1), ZO_grad_output.view(-1), dim=0)}')
                # print(f'{torch.linalg.norm(ZO_grad_output.view(-1)) / torch.linalg.norm(grad_output[0].view(-1))}')
            return [ZO_grad_output, None]
        return bwd_pre_hook

    @staticmethod
    def create_fwd_hook_get_param_grad(ZO_grad_output, debug=False):
        def fwd_hook(module, input, output):
            # module.weight.grad = torch.einsum('bsi,bsj->ij', ZO_grad_output, input[0]) * module.scale_bias.view(-1,1)
            module.weight.grad = torch.einsum('bsi,bsj->ij', ZO_grad_output, input[0] / input[1])
        return fwd_hook







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
        if isinstance(module, (RealQuantLinear,)):
            module.wp_add_perturbation(sigma=sigma, rand_gen_fn=rand_gen_fn)
        # elif isinstance(module, (nn.LayerNorm, nn.Linear)):
        # elif isinstance(module, (nn.Linear)):
        #     default_wp_add_perturbation(module, sigma*fp_scale, rand_gen_fn)

    return model

def efficient_real_quant_gen_grad(model: nn.Module, loss_diff_list, seed_list, lr=None):
    fp_scale = 1e-3
    # fp_scale = 1
    
    for i in range(len(loss_diff_list)):
        loss_diff = loss_diff_list[i]
        torch.manual_seed(seed_list[i])
        rand_gen_fn = build_rand_gen_fn(sample_method='bernoulli', device='cuda')
        
        for name, module in model.named_modules():
            if isinstance(module, (RealQuantLinear,)):
                module.wp_gen_grad(loss_diff, rand_gen_fn, lr)
            # elif isinstance(module, (nn.LayerNorm, nn.Linear)):
            # elif isinstance(module, (nn.Linear)):
            #     default_wp_gen_grad(module, loss_diff, rand_gen_fn, lr / fp_scale)
        
def default_create_fwd_hook_add_perturbation(seed, sigma, rand_gen_fn):
    def fwd_hook(module, input, output):
        # input is a tuple
        # module.in_value = input[0].detach().clone()
        # output is a tensor. inplace & return modifiled output both work
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        perturbation = rand_gen_fn(output.shape)
        torch.set_rng_state(state)
        module.perturbation = perturbation
        
        # output += sigma * perturbation
        return output + sigma * perturbation
    return fwd_hook

# def np_grad_estimation


# def 

def replace_Quant_with_RealQuant(net):
    for m in net.modules():
        to_update_dict = {}
        to_update_embedding_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, QuantLinear):
                to_update_dict[name] = sub_module
            # elif isinstance(sub_module, QuantEmbedding):
            #     to_update_embedding_dict[name] = sub_module
                
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
        
        # for name, sub_module in to_update_embedding_dict.items():
        #     m._modules[name] = RealQuantEmbedding(
        #         sub_module.num_,
        #         sub_module.dim,
        #         sub_module.weight,
        #         sub_module.padding_idx,
        #         sub_module.max_norm,
        #         sub_module.norm_type,
        #         sub_module.scale_grad_by_freq,
        #         sub_module.sparse,
        #         None,
        #         sub_module.weight_bit,
        #         sub_module.momentum,
        #         sub_module.quant_mode,
        #     )
        #     m._modules[name].weight.requires_grad = sub_module.weight.requires_grad

def build_rand_gen_fn(sample_method, device, sampler=None):
    def _rand_gen_fn(shape):
        if sample_method == 'bernoulli':
            ### Rademacher
            sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
        else:
            return NotImplementedError('Unlnown sample method', sample_method)
        
        return sample
    return _rand_gen_fn

from timm.utils.clip_grad import dispatch_clip_grad
class RealQuant_Scaler:

    def __init__(self, model):
        self.model = model
    
    @staticmethod
    def pre_step(model):
        # add a pre_step method to scale the gradient, since sometimes we need information from the model,
        # but not only parameters.
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    m.bias.grad.data = m.bias.grad.data / m.scale_bias ** 2
                if m.weight.grad is not None:
                    m.weight.grad.data = m.weight.grad.data / m.scale_w.view(-1,1) ** 2
    
    @staticmethod
    def post_step(model):
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    m.bias.data = m.bias.data.round().clamp(- 2 ** (m.bias_bit - 1), 2 ** (m.bias_bit - 1) - 1)
                if m.weight.grad is not None:
                    m.weight.data = m.weight.data.round().clamp(- 2 ** (m.weight_bit - 1), 2 ** (m.weight_bit - 1) - 1)

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        loss.backward(create_graph=create_graph)
            
        self.pre_step(self.model)
        
        optimizer.step()
        
        self.post_step(self.model)
        
        optimizer.zero_grad()
            
    # def state_dict(self):
    #     return self._scaler.state_dict()

    # def load_state_dict(self, state_dict):
    #     self._scaler.load_state_dict(state_dict)
        
# class RealQuant_NativeScaler:
#     state_dict_key = "amp_scaler"

#     def __init__(self, model):
#         self._scaler = torch.cuda.amp.GradScaler()
#         self.model = model
    
#     @staticmethod
#     def pre_step(model):
#         # add a pre_step method to scale the gradient, since sometimes we need information from the model,
#         # but not only parameters.
#         for m in model.modules():
#             if isinstance(m, RealQuantLinear):
#                 if m.bias.grad is not None:
#                     m.bias.grad.data = m.bias.grad.data / m.scale_bias ** 2
#                 if m.weight.grad is not None:
#                     m.weight.grad.data = m.weight.grad.data / m.scale_w.view(-1,1) ** 2
    
#     @staticmethod
#     def post_step(model):
#         for m in model.modules():
#             if isinstance(m, RealQuantLinear):
#                 if m.bias.grad is not None:
#                     m.bias.data = m.bias.data.round().clamp(- 2 ** (m.bias_bit - 1), 2 ** (m.bias_bit - 1) - 1)
#                 if m.weight.grad is not None:
#                     m.weight.data = m.weight.data.round().clamp(- 2 ** (m.weight_bit - 1), 2 ** (m.weight_bit - 1) - 1)

#     def __call__(
#             self,
#             loss,
#             optimizer,
#             clip_grad=None,
#             clip_mode='norm',
#             parameters=None,
#             create_graph=False,
#             need_update=True,
#     ):
#         self._scaler.scale(loss).backward(create_graph=create_graph)
#         if need_update:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            
#             self.pre_step(self.model)
            
#             self._scaler.step(optimizer)
            
#             self.post_step(self.model)
            
#             self._scaler.update()
            
#     def state_dict(self):
#         return self._scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict)

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, model):
        self._scaler = torch.cuda.amp.GradScaler()
        self.model = model

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
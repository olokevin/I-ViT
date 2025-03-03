import torch
import torch.nn as nn

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC

### Model specific utils ###
# from .ZO_fwd_utils import trainable_layers_dict, get_iterable_block_name, ZO_pre_block_forward, ZO_post_block_forward

from quant.real_quant import RealQuantLinear

def create_trainable_layers_list(layer_list, trainable_layers_dict):
    if isinstance(layer_list, str):
        return trainable_layers_dict[layer_list]
    elif isinstance(layer_list, list):
        opt_layers = []
        for layer_str in layer_list:
            opt_layers.append(trainable_layers_dict[layer_str])
        return tuple(opt_layers)
    else:
        raise (ValueError("opt_layers_strs should either be a string of a list of strings"))

def fwd_hook_get_output_shape(module, input, output):
    module.output_shape = output.shape

def build_ZO_Estim(config, model):
    if config.name == 'ZO_Estim_MC':
        ### split model
        ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        split_modules_list = split_model(model, ZO_iterable_block_name)

        splited_param_list = None
        splited_layer_list = None

        ### Param perturb 
        if config.param_perturb_block_idx_list is not None:
            splited_param_list = []
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    splited_param_list.append(SplitedParam(idx=-1, name=param_name, layer=None, param=param)) 
            
            # splited_layer_list = []
            # for layer_name, layer in model.named_modules():
            #     if type(layer) in (RealQuantLinear,):
            #         if layer.weight.requires_grad:
            #             splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer, mode='param'))
        
        # if config.param_perturb_block_idx_list is not None:
        #     if config.param_perturb_block_idx_list == 'all':
        #         param_perturb_block_idx_list = list(range(len(split_modules_list)))
        #     else:
        #         param_perturb_block_idx_list = config.param_perturb_block_idx_list
            
        #     splited_param_list = []
            
        #     if config.en_partial_forward:                        
        #         for block_idx in param_perturb_block_idx_list:
        #             if block_idx < 0:
        #                 block_idx = len(split_modules_list) + block_idx
        #             block = split_modules_list[block_idx]
                    
        #             for layer_name, layer in block.named_modules():
        #                 if isinstance(layer, (TensorizedLinear, nn.Linear, nn.Conv2d)):
        #                     for param_name, param in layer.named_parameters():
        #                         if param.requires_grad:
        #                             splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param))  
        #     else:
        #         for param_name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 splited_param_list.append(SplitedParam(idx=-1, name=param_name, layer=None, param=param)) 
                
        ### Actv perturb 
        if config.actv_perturb_block_idx_list is not None:
            splited_layer_list = []
            
            for layer_name, layer in model.named_modules():
                if type(layer) in (RealQuantLinear,):
                    if layer.weight.requires_grad:
                        splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer))

        #     if config.actv_perturb_block_idx_list == 'all':
        #         actv_perturb_block_idx_list = list(range(len(split_modules_list)))
        #     else:
        #         actv_perturb_block_idx_list = config.actv_perturb_block_idx_list
            
        #     # if config.ZO_trainable_layers_list is not None:
        #     if hasattr(model, 'ZO_trainable_layers_dict'):
        #         ZO_trainable_layers_dict = model.ZO_trainable_layers_dict
        #         ZO_trainable_layers_list = create_trainable_layers_list(config.ZO_trainable_layers_list, ZO_trainable_layers_dict)
        #     else:
        #         ZO_trainable_layers_list = None

            
        #     ### model specified actv perturb
        #     if hasattr(model, 'ZO_trainable_blocks_name_idx'):
        #         for name, idx in model.ZO_trainable_blocks_name_idx.items():
        #             if idx == -1:
        #                 idx = len(split_modules_list)
        #             splited_layer_list.append(SplitedLayer(idx=idx, name=name, layer=getattr(model, name)))
            
        #     # if 'ATIS' in config.obj_fn_type:
        #     #     splited_layer_list.append(SplitedLayer(idx=0, name='embedding', layer=model.embedding))
            
        #     if config.en_partial_forward:
                
        #         for block_idx in actv_perturb_block_idx_list:
        #             if block_idx < 0:
        #                 block_idx = len(split_modules_list) + block_idx
        #             block = split_modules_list[block_idx]
        #             if ZO_trainable_layers_list is not None:
        #                 if type(block) in ZO_trainable_layers_list:
        #                     splited_layer_list.append(SplitedLayer(idx=block_idx, name=f'{ZO_iterable_block_name}.{block_idx}', layer=block))
        #                 else:
        #                     for name, layer in block.named_children():
        #                         if type(layer) in ZO_trainable_layers_list:
        #                             splited_layer_list.append(SplitedLayer(idx=block_idx, name=f'{ZO_iterable_block_name}.{block_idx}.{name}', layer=layer))
            
        #     else:
        #         for layer_name, layer in model.named_modules():
        #             if type(layer) in (TensorizedLinear, nn.Linear, nn.Conv2d):
        #                 splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer))

        #     # if 'ATIS' in config.obj_fn_type:
        #     #     splited_layer_list.append(SplitedLayer(idx=len(split_modules_list), name='classifier', layer=model.classifier))
        #     #     splited_layer_list.append(SplitedLayer(idx=len(split_modules_list), name='slot_classifier', layer=model.slot_classifier))
                            
        # if splited_param_list is not None:
        #     for splited_param in splited_param_list:
        #         print('param', splited_param.name)
            
        # if hasattr(model, 'ZO_trainable_layers_list_wp'):
        #     ZO_trainable_layers_list_wp = model.ZO_trainable_layers_list_wp
        # else:
        #     ZO_trainable_layers_list_wp = None
        
        if splited_param_list is not None:
            for splited_param in splited_param_list:
                print('param', splited_param.name)
        
        if splited_layer_list is not None:
            for splited_layer in splited_layer_list:
                print('layer', splited_layer.name)

            # if ZO_trainable_layers_list_wp is not None:
            #     for splited_layer in splited_layer_list:
            #         if isinstance(splited_layer.layer, ZO_trainable_layers_list_wp):
            #             splited_layer.mode = 'param'
        
        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn_type = config.obj_fn_type,
            splited_param_list = splited_param_list,
            splited_layer_list = splited_layer_list,
            
            config = config,
        )
        return ZO_Estim
    else:
        return NotImplementedError

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_layerwise':
        obj_fn = build_obj_fn_classifier_layerwise(**kwargs)
    else:
        raise NotImplementedError
    return obj_fn
  
def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn(return_loss_reduction='mean'):
        y = model(data)
        # return y, criterion(y, target)
      
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            return y, criterion(y, target)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            loss = criterion(y, target)
            criterion.reduction = 'mean'
            return y, loss
    
    return _obj_fn

def build_obj_fn_classifier_layerwise(data, target, model, criterion):
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                y = ZO_pre_block_forward(y)
        else:
            assert input is not None
            y = input
        
        # if input is not None:
        #     y = input
        # else:
        #     assert starting_idx == 0
        #     y = data
        #     ### ZO_pre_block_forward when start from image input
        #     if ZO_pre_block_forward is not None:
        #         y = ZO_pre_block_forward(y)
        
        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            y = split_modules_list[i](y)
            if detach_idx is not None and i == detach_idx:
                y = y.detach()
                y.requires_grad = True
           
        if return_loss_reduction == 'no_loss':
            return y
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    y = ZO_post_block_forward(y)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                return y, criterion(y, target)
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                loss = criterion(y, target)
                criterion.reduction = 'mean'
                return y, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn
import torch
from ZO.real_quant import RealQuantLinear

class SGDInt(torch.optim.SGD):
    @staticmethod
    def post_step(model):
        
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    m.bias.data = m.bias.data.round()
                if m.weight.grad is not None:
                    m.weight.data = m.weight.data.round()

class SGDScale(torch.optim.SGD):
    @staticmethod
    def pre_step(model):
        # add a pre_step method to scale the gradient, since sometimes we need information from the model,
        # but not only parameters.
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    # m.bias.grad.data = m.bias.grad.data / (m.effective_scale.data * m.scale_y) ** 2
                    m.bias.grad.data = m.bias.grad.data / (m.scale_x * m.scale_w) ** 2
                if m.weight.grad is not None:
                    # scale_w = m.effective_scale.data * m.scale_y / m.scale_x
                    scale_w = m.scale_w
                    m.weight.grad.data = m.weight.grad.data / scale_w.view(-1, 1, 1, 1) ** 2

class SGDScaleInt(torch.optim.SGD):
    @staticmethod
    def pre_step(model):
        # add a pre_step method to scale the gradient, since sometimes we need information from the model,
        # but not only parameters.
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    m.bias.grad.data = m.bias.grad.data / m.scale_bias ** 2
                if m.weight.grad is not None:
                    m.weight.grad.data = m.weight.grad.data / m.scale_w ** 2
    
    @staticmethod
    def post_step(model):
        for m in model.modules():
            if isinstance(m, RealQuantLinear):
                if m.bias.grad is not None:
                    m.bias.data = m.bias.data.round().clamp(- 2 ** (m.bias_bit - 1), 2 ** (m.bias_bit - 1) - 1)
                if m.weight.grad is not None:
                    m.weight.data = m.weight.data.round().clamp(- 2 ** (m.weight_bit - 1), 2 ** (m.weight_bit - 1) - 1)
from torch import autograd
import torch


class TrinaryStep(autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        output[input > 1] = 1
        output[input < -1] = -1
        output[torch.logical_and(-0.5 < input, input < 0.5)] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
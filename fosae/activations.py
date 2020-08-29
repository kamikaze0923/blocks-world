from torch import autograd
import torch


class TrinaryStep(autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.zeros(size=input.size(), device=input.device)
        output[input > 1] = 1
        output[input < -1] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
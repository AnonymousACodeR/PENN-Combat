import torch
from torch import nn


class PESymetryMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMean, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.rest(x_mean)
        x = self.diagonal(x)
        x = x + x_mean
        return x


class PESymetryMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMax, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = x.max(0, keepdim=True)
        x_max = self.rest(x_max)
        x = self.diagonal(x)
        x = x + x_max
        return x


class PESymetryMeanMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMeanMax, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.max = nn.Linear(in_dim, out_dim, bias=False)
        self.mean = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = x.max(0, keepdim=True)
        x_max = self.max(x_max)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.mean(x_mean)
        x = self.diagonal(x)
        x = x + x_max + x_mean
        return x


class PESymetryMeanAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMeanAct, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.rest(x_mean)
        x_mean = nn.ELU()(x_mean)
        x = self.diagonal(x)
        # x = nn.ELU()(x)
        x = x + x_mean
        return x


class RPESymetryMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(RPESymetryMean, self).__init__()
        self.diagonal = nn.GRUCell(in_dim, out_dim)
        self.rest = nn.GRUCell(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # x_mean = x.mean(0, keepdim=True)
        output = []
        for i, j in zip(x.view(-1, x.shape[-2], x.shape[-1]), h.view(-1, h.shape[-2], h.shape[-1])):
            # print(f"Shape i: {i.shape}, shape j: {j.shape}")
            x_mean = i.mean(-2, keepdim=True)
            h_mean = j.mean(-2, keepdim=True)
            h_mean = self.rest(x_mean, h_mean)
            h_out = self.diagonal(i, j)
            h_out = h_out + h_mean
            output.append(h_out)
        output = torch.stack(output, dim=0)
        return output.view(h.shape)

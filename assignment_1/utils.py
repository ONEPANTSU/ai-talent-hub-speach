import torch.nn as nn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

from dataset import YesNoDataset
from сnn import SimpleCNN


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: SimpleCNN):
    macs, params = get_model_complexity_info(
        model, (16000,), as_strings=False, print_per_layer_stat=False
    )
    return macs, params


def print_layer_breakdown(model: SimpleCNN):
    print(f"  {'Layer':<35} {'groups':>6} {'params':>8}   formula")
    print(f"  {'-'*35} {'-'*6} {'-'*8}   {'-'*25}")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d):
            g = m.groups
            p = sum(p.numel() for p in m.parameters())
            formula = f"{m.in_channels}*{m.out_channels}*{m.kernel_size[0]}/{g} = {p}"
            print(f"  {name:<35} {g:>6} {p:>8,}   {formula}")
    total = count_parameters(model)
    print(f"  {'TOTAL':<35} {'':>6} {total:>8,}")


_cached_datasets = {}


def make_dataloaders(batch_size: int = 64, num_workers: int = 0):
    for subset in ("training", "validation", "testing"):
        if subset not in _cached_datasets:
            _cached_datasets[subset] = YesNoDataset(subset)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(_cached_datasets["training"], shuffle=True, **common),
        DataLoader(_cached_datasets["validation"], shuffle=False, **common),
        DataLoader(_cached_datasets["testing"], shuffle=False, **common),
    )
import torch
from torch import optim, nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindBlock
# 定义adapter网络结构
class adapter(nn.Module):
    def __init__(self, in_features,middle_features):
        super().__init__()
        self.A = nn.Linear(in_features, middle_features)
        self.B = nn.Linear(middle_features, in_features)

    def forward(self, x):
        return self.B(torch.nn.functional.gelu(self.A(x)))


def apply_adapter(model, middle_features=8):
    for name, module in model.named_modules():
        if isinstance(module, MiniMindBlock):
            adapter1 = adapter(module.hidden_size, middle_features).to(model.device)
            adapter2 = adapter(module.hidden_size, middle_features).to(model.device)
            module.adapter1 = adapter1
            module.adapter2 = adapter2
            


def load_adapter(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'adapter1'):
            adapter1_state = {k.replace(f'{name}.adapter1.', ''): v for k, v in state_dict.items() if f'{name}.adapter1.' in k}
            adapter2_state = {k.replace(f'{name}.adapter2.', ''): v for k, v in state_dict.items() if f'{name}.adapter2.' in k}
            module.adapter1.load_state_dict(adapter1_state)
            module.adapter2.load_state_dict(adapter2_state)


def save_adapter(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'adapter1'):
            adapter1_state = {f'{name}.adapter1.{k}': v for k, v in module.adapter1.state_dict().items()}
            adapter2_state = {f'{name}.adapter2.{k}': v for k, v in module.adapter2.state_dict().items()}
            state_dict.update(adapter1_state)
            state_dict.update(adapter2_state)
    torch.save(state_dict, path)

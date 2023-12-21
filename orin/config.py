# code for default model settings

import torch

# TODO: add more global config in here
class OPTConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float32
    # mixed precision config
    
    # int8() replacement setting
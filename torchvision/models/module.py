import torch
from torch import nn


class StopForward(Exception):
    """
    Used to signal the end of the forward
    """
    pass


class VisionModule(nn.Module):
    """
    This is a replacement of nn.Module that adds support
    for early forward exit, and also stores intermediate
    outputs if requested
    """
    def __init__(self, *args, **kwargs):
        super(VisionModule, self).__init__(*args, **kwargs)
        self._save_output = False
        self._is_last = False
        
    def __call__(self, *args, **kwargs):
        output = super(VisionModule, self).__call__(*args, **kwargs)
        if self._save_output:
            self._output = output
            if self._is_last:
                raise StopForward()
        return output


class VisionModel(VisionModule):
    def __call__(self, *args, **kwargs):
        try:
            _ = super(VisionModel, self).__call__(*args, **kwargs)
        except StopForward:
            pass
        outputs = {}
        for name, module in self.named_modules():
            if module._save_output:
                outputs[name] = module._output
                del module._output
        return outputs

# create copies of all modules in nn so that they follow this
# structure
for name in nn.modules.__all__:
    if issubclass(nn.modules.__dict__[name], nn.Module):
        globals()[name] = type(name, (VisionModule, nn.modules.__dict__[name]), {})

from torch.nn import init

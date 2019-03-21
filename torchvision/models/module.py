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
            normal_output = super(VisionModel, self).__call__(*args, **kwargs)
        except StopForward:
            pass
        outputs = {}
        for name, module in self.named_modules():
            if module._save_output:
                outputs[name] = module._output
                del module._output
        if not outputs:
            return normal_output
        return outputs


    def setup_for_return_values(self, return_layers=None):
        if not return_layers:
            return
        to_be_returned = []
        for name, module in self.named_modules():
            module._save_output = False
            module._is_last = False
            if name in return_layers:
                module._save_output = True
                to_be_returned.append(name)
                # TODO needs better way of enforcing
                # which one is the last layer to be
                # computed
                if name == return_layers[-1]:
                    module._is_last = True
        assert set(to_be_returned) == set(return_layers)

# create copies of all modules in nn so that they follow this
# structure
for name in nn.modules.__all__:
    if issubclass(nn.modules.__dict__[name], nn.Module):
        globals()[name] = type(name, (VisionModule, nn.modules.__dict__[name]), {})

from torch.nn import init

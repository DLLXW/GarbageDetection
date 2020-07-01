import torchvision.models as models
from collections import OrderedDict
import torch.nn as nn
import torch
from ssd.modeling import registry
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

return_layers = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
backbone = models.resnet101(pretrained=True)
body = IntermediateLayerGetter(backbone, return_layers)
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.body=body
    def forward(self, x):
        features = self.body(x)
        # return them as a tuple
        return tuple(features.values())
#model=resnet()
#x=torch.randn(1,3,512,512)
#out=model(x)
#print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
@registry.BACKBONES.register('res_backbone')
def res_backbone(cfg, pretrained=True):
    model = resnet()
    return model

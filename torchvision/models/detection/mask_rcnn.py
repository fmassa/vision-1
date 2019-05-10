import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .poolers import Pooler
from .roi_heads import RoIHeads

from .._utils import IntermediateLayerGetter
from . import fpn as fpn_module

from torchvision.ops import misc as misc_nn_ops

from maskrcnn_benchmark.layers import FrozenBatchNorm2d


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

def load_resnet_c2_format(f):
    from maskrcnn_benchmark.utils.c2_model_loading import _load_c2_pickled_weights, _C2_STAGE_NAMES, _rename_weights_for_resnet
    state_dict = _load_c2_pickled_weights(f)
    conv_body = "R-50-FPN"
    arch = conv_body.replace("-C4", "").replace("-C5", "").replace("-FPN", "")
    arch = arch.replace("-RETINANET", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    return state_dict

def build_resnet_fpn_backbone(backbone_name):
    from .. import resnet
    from .._utils import IntermediateLayerGetter
    from . import fpn as fpn_module

    backbone = resnet.__dict__[backbone_name](
        # pretrained=False,
        pretrained=True,
        norm_layer=FrozenBatchNorm2d)

    # del backbone.avgpool
    # del backbone.fc

    if False:
        state_dict = load_resnet_c2_format('/private/home/fmassa/.torch/models/R-50.pkl')
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict
        load_state_dict(backbone, state_dict)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    for name, parameter in body.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    in_channels_stage2 = 256  # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = 256  # cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

def build_backbone_with_fpn(backbone, return_layers, in_channels_list, out_channels):
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

def build_resnet_fpn_backbone_(backbone_name):
    from .. import resnet
    backbone = resnet.__dict__[backbone_name](
        # pretrained=False,
        pretrained=True,
        norm_layer=FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = 256
    out_channels = 256
    in_channels_list=[
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    return build_backbone_with_fpn(backbone, return_layers, in_channels_list, out_channels)


class FasterRCNN(GeneralizedRCNN):
    pass



class MaskRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes,
                 #
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 #
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 #
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 mask_discretization_size=28):

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = Pooler(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        if mask_roi_pool is None:
            mask_roi_pool = Pooler(
                featmap_names=[0, 1, 2, 3],
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_dim_reduced = 256  # == mask_layers[-1]
            mask_predictor = MaskRCNNC4Predictor(out_channels, mask_dim_reduced, num_classes)

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            # Mask
            mask_roi_pool,
            mask_head,
            mask_predictor,
            mask_discretization_size
            )

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads)



def maskrcnn_resnet50_fpn(pretrained=False, num_classes=81, **kwargs):
    backbone = build_resnet_fpn_backbone('resnet50')
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        pass
    return model


class TwoMLPHead(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            conv = misc_nn_ops.Conv2d(next_feature, layer_features, kernel_size=3,
                    stride=1, padding=dilation, dilation=dilation)
            d["mask_fcn{}".format(layer_idx)] = conv
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNC4Predictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNC4Predictor, self).__init__(OrderedDict([
            ("conv5_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


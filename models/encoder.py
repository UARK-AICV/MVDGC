import torch
import torch.nn as nn
import torchvision

# from efficientnet_pytorch import EfficientNet
from timm.utils.model import freeze_batch_norm_2d


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        freeze_bn(resnet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.inplanes = 2048
        self.deconv_with_bias = False
        
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # resnet = torchvision.models.resnet50(pretrained=True)
        # freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # used for deconv layers
        NUM_DECONV_LAYERS = 3
        NUM_DECONV_FILTERS = [256,256,256]
        NUM_DECONV_KERNELS = [4,4,4]
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV_LAYERS,
            NUM_DECONV_FILTERS,
            NUM_DECONV_KERNELS,
        )
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
        
    def forward(self, x, use_feat_level=[0, 1, 2]):
        x = self.layer0(x)  # torch.Size([1, 64, 192, 320])
        x = self.layer1(x)  # torch.Size([1, 256, 192, 320])
        x = self.layer2(x)  # torch.Size([1, 512, 96, 160])
        x = self.layer3(x)  # torch.Size([1, 1024, 48, 80])
        x = self.layer4(x)  # torch.Size([1, 2048, 23, 40])
        
        interm_feat = []
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                interm_feat.append(x)

        return [f for (i, f) in enumerate(interm_feat) if i in use_feat_level]

class Encoder_res18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.inplanes = 512
        self.deconv_with_bias = False
        
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # used for deconv layers
        NUM_DECONV_LAYERS = 3
        NUM_DECONV_FILTERS = [256,256,256]
        NUM_DECONV_KERNELS = [4,4,4]
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV_LAYERS,
            NUM_DECONV_FILTERS,
            NUM_DECONV_KERNELS,
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, use_feat_level=[0, 1, 2]):
        # import ipdb; ipdb.set_trace()
        x = self.layer0(x)  # torch.Size([1, 64, 180, 320])
        x = self.layer1(x)  # torch.Size([1, 64, 180, 320])
        x = self.layer2(x)  # torch.Size([1, 128, 90, 160])
        x = self.layer3(x)  # torch.Size([1, 256, 45, 80])
        x = self.layer4(x)  # torch.Size([1, 512, 23, 40])
        
        interm_feat = []
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                interm_feat.append(x)

        return [f for (i, f) in enumerate(interm_feat) if i in use_feat_level]
    

class Encoder_fasterrcnn_resnet50_fpn(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.inplanes = 2048
        self.deconv_with_bias = False
        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        freeze_bn(model)
        
        self.backbone = model.backbone.body
        self.fpn_layers = model.backbone.fpn
        
    def forward(self, x, use_feat_level=[3, 2, 1, 0]):
        x = self.backbone(x)
        x = self.fpn_layers(x)
        
        interm_feat = []
        for lvl in use_feat_level:
            interm_feat.append(x[str(lvl)])

        return [f for (i, f) in enumerate(interm_feat) if i in use_feat_level]


class Encoder_fasterrcnn_resnet50_fpn_trainable(nn.Module):
    def __init__(self, C, num_classes, input_size):
        super().__init__()
        self.C = C
        self.inplanes = 2048
        self.deconv_with_bias = False
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        self.model.transform = GeneralizedRCNNTransform(
            min_size=input_size[0],
            max_size=input_size[1],
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            fixed_size=input_size  # This ensures fixed output size
        )
        
        # import ipdb; ipdb.set_trace()
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # self.backbone_features = {}
        self.fpn_features = {}
        
        # Register hooks to capture features
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        # Hook for backbone features (ResNet layers)
        # def get_backbone_hook(name):
        #     def hook(module, input, output):
        #         self.backbone_features[name] = output
        #     return hook
        
        # # Register hooks on ResNet backbone layers
        # backbone = self.model.backbone.body
        # backbone.layer1.register_forward_hook(get_backbone_hook('layer1'))
        # backbone.layer2.register_forward_hook(get_backbone_hook('layer2'))
        # backbone.layer3.register_forward_hook(get_backbone_hook('layer3'))
        # backbone.layer4.register_forward_hook(get_backbone_hook('layer4'))
        
        # Hook for FPN features
        def get_fpn_hook(name):
            def hook(module, input, output):
                if isinstance(output, dict):
                    self.fpn_features[name] = output
                else:
                    self.fpn_features[name] = output
            return hook
        
        # FPN outputs
        self.model.backbone.fpn.register_forward_hook(get_fpn_hook('fpn'))
        
    def forward(self, x, targets, use_feat_level=[2, 1, 0]):
        self.fpn_features = {}
        loss_dict = self.model(x, targets)
        features = self.fpn_features
        
        return loss_dict, features


        
# fake_ing = torch.rand(1,3,768,1280)
# encoder = Encoder_res18(256)
# output=encoder(fake_ing)

# output[0].shape     torch.Size([1, 256, 48, 80])
# output[1].shape     torch.Size([1, 256, 96, 160])
# output[2].shape     torch.Size([1, 256, 192, 320])



# fake_ing = torch.rand(1,3,768,1280)
# encoder = Encoder_res50(256)
# output=encoder(fake_ing)

# output[0].shape torch.Size([1, 256, 48, 80])
# output[1].shape torch.Size([1, 256, 96, 160])
# output[2].shape torch.Size([1, 256, 192, 320])


# fake_ing = torch.rand(1,3,768,1280)
# encoder = Encoder_fasterrcnn_resnet50_fpn_trainable(256,2)
# output=encoder(fake_ing)

# import ipdb; ipdb.set_trace()




# pretrain_dict = torch.load('/home/thinhphan/april18_bevgeo/eva02_L_coco_det_sys_o365.pth', map_location=torch.device('cpu'))
# pretrain_dict = pretrain_dict["model"]
# print(pretrain_dict.keys())
# remapped_dict = {}
# for k,v in pretrain_dict.items():
#     if "backbone.net" in k:
#         remapped_dict[k.replace("backbone.net.", "img_backbone.")] = v
#     if "backbone.simfp" in k:
#         remapped_dict[k.replace("backbone.", "img_backbone.adapter.")] = v
# torch.save(remapped_dict,'/home/thinhphan/april18_bevgeo/eva02_L_coco_det_sys_o365_remapped.pth')



# fake_ing = torch.rand(1,3,768,1280)
# pretrain_dict = torch.load('/home/thinhphan/april18_bevgeo/eva02_L_coco_det_sys_o365.pth', map_location=torch.device('cpu'))
# output=pretrain_dict(fake_ing)
# import ipdb; ipdb.set_trace()
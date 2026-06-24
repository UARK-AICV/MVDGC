import torch
import torch.nn as nn
from mmdet.models import build_backbone, build_neck

class CoDETR(nn.Module):
    def __init__(self):
        super().__init__()
        window_block_indexes = (
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
        )
        residual_block_indexes = []
        backbone_config=dict(
                type='ViT',
                img_size=640,
                pretrain_img_size=512,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4*2/3,
                drop_path_rate=0.3,
                window_size=16,
                window_block_indexes=window_block_indexes,
                residual_block_indexes=residual_block_indexes,
                qkv_bias=True,
                use_act_checkpoint=True,
                use_lsj=True,
                init_cfg=None)
        neck_config=dict(        
                type='SFP',
                in_channels=[1024],        
                out_channels=256,
                num_outs=5,
                use_p2=True,
                use_act_checkpoint=False)
        self.backbone = build_backbone(backbone_config)
        self.neck = build_neck(neck_config)

        pretrained = "./vit.pth"
        checkpoint = torch.load(pretrained, map_location='cpu')
        checkpoint = checkpoint['state_dict']

        #     # #BACKBONE
        bb_weights = {k.replace("backbone.", ""): v
                        for k, v in checkpoint.items()
                        if k.startswith("backbone.")}

        missing, unexpected = self.backbone.load_state_dict(bb_weights, strict=False)
        print("-----------------BACKBONE--------------")
        
        print(missing)
        print("\n")
        print(unexpected)
        print("-----------------BACKBONE--------------")
        neck_weights = {k.replace("neck.", ""): v
                        for k, v in checkpoint.items()
                        if k.startswith("neck.")}
        missing, unexpected = self.neck.load_state_dict(neck_weights, strict=False)
        print("-----------------NECK--------------")
        print(missing)
        print("\n")
        print(unexpected)
        print("-----------------NECK--------------")

    def forward(self, imgs):
        feats = self.backbone(imgs)
        feats = self.neck(feats)
        return feats


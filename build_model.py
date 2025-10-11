from utils import log_model_source, save_macs_params_count

import torch
from timm.models.vision_transformer import VisionTransformer


def build_model(args):
    if args.model == "vit":
    
        model = VisionTransformer(
            img_size=args.image_size,
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=args.num_classes,         
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.n_head,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),       
        )

    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}\n")

    try:
        from fvcore.nn import FlopCountAnalysis
        model.training = False
        input = torch.randn(1, 3, args.image_size[0], args.image_size[1])
        flops = FlopCountAnalysis(model, input)
        print(f"Total MACs Estimate (fvcore): {flops.total()}")
    except:
        print("FLOPs estimator failed")
        pass

    log_model_source(model, save_dir=f"expt_logs/{args.expt_name}/model_snapshot")
    save_macs_params_count(sum(p.numel() for p in model.parameters()), flops.total(), f"expt_logs/{args.expt_name}")
    
    return model

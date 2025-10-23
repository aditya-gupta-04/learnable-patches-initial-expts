from utils import log_model_source, save_macs_params_count

import torch
import timm
from timm.models.vision_transformer import VisionTransformer
from embedding import Embeddings

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

    if args.model == "vit-b" and args.learn_patch_boundaries:
        model = timm.create_model('vit_base_patch16_224', pretrained=True)  
        for p in model.parameters():
            p.requires_grad = False      

    if args.learn_patch_boundaries:
        # Swapping the default patch embedding with our custom one
        model.patch_embed = Embeddings(img_size=args.image_size, patch_size=args.patch_size, embed_dim=model.embed_dim, encoder_depth = args.encoder_depth, ratio_loss_N=args.ratio_loss_N)
        model.pos_embed = None # Disables the positional embeddings in the timm model

        print("Initialized model with learnable patch boundaries, swapped original patch_embed and disabled original pos_embed")
        print(f"Ratio Loss N : {"No Ratio Loss" if args.ratio_loss_N == 0 else args.ratio_loss_N} | Ratio Loss weighted by alpha = {args.ratio_loss_alpha}")

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

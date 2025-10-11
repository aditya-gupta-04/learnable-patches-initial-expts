# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

__all__ = [
    "window_partition",
    "window_unpartition",
    "add_decomposed_rel_pos",
    "get_abs_pos",
    "PatchEmbed",
]

import os
import inspect
import json

def get_self_attributes(mod):
    attrs = {}
    for key, val in vars(mod).items():
        if key[0] != "_":
            try:
                json.dumps(val)
                attrs[key] = val
            except (TypeError, OverflowError):
                attrs[key] = str(val)
    return attrs

def log_model_source(model, save_dir="logs/model_snapshot", name="model"):
    os.makedirs(save_dir, exist_ok=True)

    module_attr_log = {}
    seen_classes = set()

    for module_path, mod in model.named_modules():
        cls = mod.__class__

        if cls.__name__ in {"Conv1d", "Conv2d", "Linear", "LayerNorm", "ConvTranspose2d"}:
            continue

        # Unique module key
        module_key = f"{name}.{module_path}" if module_path else name
        module_attr_log[module_key] = get_self_attributes(mod)

        # Log class source once
        if cls not in seen_classes:
            seen_classes.add(cls)
            try:
                source = inspect.getsource(cls)
                fname = os.path.join(save_dir, f"{cls.__name__}.txt")
                with open(fname, "w") as f:
                    f.write(source)
            except (TypeError, OSError):
                pass

    with open(os.path.join(save_dir, f"{name}_self_attrs.json"), "w") as f:
        json.dump(module_attr_log, f, indent=2)

    with open(os.path.join(save_dir, f"{name}_summary.txt"), "w") as f:
        f.write(str(model))

    print(f"Logged {name} in {save_dir}")


def save_macs_params_count(param_count, macs, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, "model_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Number of model parameters: {param_count}\n")

    with open(metrics_path, "a") as f:
        f.write(f"Total MACs Estimate (fvcore): {macs}\n")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this on a login node with internet access.
Downloads ESM2 model assets and optionally TR-Rosetta/SidechainNet data,
so compute nodes can run in offline mode.
"""

import argparse
import json
from pathlib import Path


def try_download_hf_model(repo_id, out_dir):
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        return {"ok": False, "reason": f"huggingface_hub unavailable: {e}"}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=repo_id, local_dir=str(out_dir), local_dir_use_symlinks=False)
    return {"ok": True, "local_path": str(local)}


def try_download_esm_native(model_name, out_dir):
    import torch
    import esm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Triggers download into torch hub cache
    esm.pretrained.load_model_and_alphabet(model_name)

    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    model_pt = ckpt_dir / f"{model_name}.pt"
    reg_pt = ckpt_dir / f"{model_name}-contact-regression.pt"

    copied = []
    if model_pt.exists():
        dst = out_dir / model_pt.name
        dst.write_bytes(model_pt.read_bytes())
        copied.append(str(dst))
    if reg_pt.exists():
        dst = out_dir / reg_pt.name
        dst.write_bytes(reg_pt.read_bytes())
        copied.append(str(dst))

    if not copied:
        return {"ok": False, "reason": "downloaded files not found in torch hub cache"}
    return {"ok": True, "files": copied}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", type=str, default="facebook/esm2_t36_3B_UR50D")
    ap.add_argument("--esm_model_name", type=str, default="esm2_t36_3B_UR50D")
    ap.add_argument("--model_out", type=str, default="./assets/models/esm2_t36_3B_UR50D")
    ap.add_argument("--data_out", type=str, default="./assets/data/tr_rosetta")
    ap.add_argument("--download_sidechainnet", action="store_true")
    ap.add_argument("--report_out", type=str, default="./assets/offline_prepare_report.json")
    args = ap.parse_args()

    report = {"model": {}, "dataset": {}, "offline_env": {}}

    report["model"]["hf"] = try_download_hf_model(args.hf_repo, args.model_out)
    report["model"]["esm_native"] = try_download_esm_native(args.esm_model_name, args.model_out)

    data_out = Path(args.data_out)
    data_out.mkdir(parents=True, exist_ok=True)
    report["dataset"]["data_out"] = str(data_out.resolve())
    report["dataset"]["note"] = (
        "TR-Rosetta data may require manual download/license handling. "
        "Place .npz/processed files under data_out and set --esm_data_path accordingly."
    )

    if args.download_sidechainnet:
        try:
            import sidechainnet as scn

            _ = scn.load(casp_version=12, thinning=100)
            report["dataset"]["sidechainnet"] = {"ok": True, "message": "Downloaded to sidechainnet cache."}
        except Exception as e:
            report["dataset"]["sidechainnet"] = {"ok": False, "reason": str(e)}

    report["offline_env"] = {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

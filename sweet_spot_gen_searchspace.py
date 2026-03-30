import json, os, argparse, itertools
from pathlib import Path

# 代表层（早-中-晚-倒数第二-最后）
SEEDS = [42]
REP_LAYERS = [0, 8, 16]

# ESM2 v1:
# - 0..5: single positions
# - 6..11: indexed position combinations
POS_LIST = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "__combo_attn_all__",   # [q,k,v,out]
    "__combo_attn_core__",  # [q,v]
    "__combo_kv_pair__",    # [k,v]
    "__combo_mlp_only__",   # [fc1,fc2]
    "__combo_out_fc2__",    # [out,fc2]
    "__combo_full_mix__",   # [q,k,v,out,fc1,fc2]
]

# ESM2 v1:
# - 0..35: single layers
# - 36..44: indexed layer combinations
LAYER_INDEX_MAP = {
    **{i: [i] for i in range(36)},
    36: [30, 31, 32, 33, 34, 35],
    37: [18, 20, 22, 24, 26, 28, 30, 32, 34],
    38: [12, 24, 35],
    39: list(range(36)),
    40: [0, 1, 2, 3, 4, 5],
    41: [14, 15, 16, 17, 18, 19],
    42: [4, 5, 6, 16, 17, 18, 30, 31, 32],
    43: [0, 1, 2, 33, 34, 35],
    44: [20, 22, 24, 26, 28, 30, 32, 34],
}

# rank（含0对照）
RANKS = [0, 8, 16, 32, 64, 128, 256]

PRECISIONS = ["int8", "int4", "fp16"]

# 组合关系（pair）与 rank 配额（总预算=16）
PAIR_LIST = [
    ("self_attn.q_proj", "self_attn.v_proj"),
    ("self_attn.q_proj", "self_attn.o_proj"),
    ("mlp.up_proj", "mlp.down_proj")
]

BUDGET_SPLITS = [(16,0),(12,4),(8,8),(4,12),(0,16)]

def make_searchspace(pos, r, prec, layer=0, sd=42):

    here = Path(__file__).resolve().parent
    out_dir = here / "searchspace" 

    os.makedirs(out_dir, exist_ok=True)
    pos = int(pos)
    layer = int(layer)
    if pos < 0 or pos >= len(POS_LIST):
        raise ValueError(f"pos index out of range: {pos}, valid [0, {len(POS_LIST)-1}]")
    if layer not in LAYER_INDEX_MAP:
        raise ValueError("layer index out of range: "
                         f"{layer}, valid [0, 44] for ESM2 v1 (single + combo)")

    single = []
    single.append({
        "type": "single",
        "layer": layer,
        "layer_group": LAYER_INDEX_MAP[layer],  # optional metadata; runtime still keys on `layer`
        "pos": POS_LIST[pos],
        "rank": int(r),
        "precision": PRECISIONS[int(prec)],
        "seed": sd
    })
    with open(os.path.join(out_dir, "configs_single.json"), "w") as f:
        json.dump(single, f, indent=2)
    print(f"[ok] generated: {out_dir}/configs_single.json  ({len(single)} items)")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    single = []
    for layer, pos, r, sd in itertools.product(REP_LAYERS, POS_LIST, RANKS, SEEDS):
        single.append({
            "type": "single",
            "layer": layer,
            "pos": pos,
            "rank": r,
            "seed": sd
        })

    pairs = []
    for layer, (p1, p2), (r1, r2) in itertools.product(REP_LAYERS, PAIR_LIST, BUDGET_SPLITS):
        pairs.append({
            "type": "pair",
            "layer": layer,
            "pos1": p1, "rank1": r1,
            "pos2": p2, "rank2": r2,
            "seed": 42
        })

    with open(os.path.join(out_dir, "configs_single.json"), "w") as f:
        json.dump(single, f, indent=2)
    with open(os.path.join(out_dir, "configs_pair.json"), "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"[ok] generated: {out_dir}/configs_single.json  ({len(single)} items)")
    print(f"[ok] generated: {out_dir}/configs_pair.json    ({len(pairs)} items)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="searchspace")
    args = ap.parse_args()
    main(args.out_dir)
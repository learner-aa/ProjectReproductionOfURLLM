"""
DG训练数据 (train_F.txt/test_F.txt) → Pipeline预处理JSON
用法: cd enhancement/ && python src/convert_dg_data.py --all
"""
import argparse, csv, json, logging
from pathlib import Path; from collections import defaultdict
from typing import Dict, List, Any, Tuple
from data_utils import PROJECT_ROOT, PROCESSED_DIR, DG_GM_DIR, DG_AO_DIR, save_json
logger = logging.getLogger(__name__)

DATASETS = {
    "GM": {
        "dg_data_dir": DG_GM_DIR,
        "train_file": "train_F.txt", "valid_file": "valid_F.txt", "test_file": "test_F.txt",
        "domain_x": "Entertainment", "domain_y": "Education", "threshold": 71067,
        "csv_paths": [DG_GM_DIR/"item_listM_F.csv", DG_GM_DIR/"item_listG_AM_F.csv"],
        "fmt": "gm",
    },
    "AO": {
        "dg_data_dir": DG_AO_DIR,
        "train_file": "train_F2.txt", "valid_file": "valid_F2.txt", "test_file": "test_F2.txt",
        "domain_x": "Entertainment", "domain_y": "Education", "threshold": 18639,
        "csv_paths": [DG_AO_DIR/"item_listA_F.csv", DG_AO_DIR/"item_listO_AA_F.csv"],
        "fmt": "ao",
    },
}

def build_mapping(csv_paths):
    m = {}
    for p in csv_paths:
        with open(p, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3: continue
                try: m[int(row[2].strip())] = row[0].strip()
                except: pass
    return m

def build_meta(csv_paths, dx, dy, th):
    meta = {}; mid = len(csv_paths)//2
    for i, p in enumerate(csv_paths):
        dom = dx if i<mid else dy
        with open(p,"r",encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3: continue
                meta[row[0].strip()] = {"title":row[1].strip(),"description":"","category":[],"brand":"","domain":dom}
    return meta

def parse_gm(line):
    parts = line.strip().split("\t")
    if len(parts)<2: return None,[]
    uid, ints = parts[0], []
    for t in parts[1:]:
        if not t.strip(): continue
        s = t.split("|")
        if len(s)<3: continue
        try: ints.append({"dg":int(s[0]),"ts":int(s[2])})
        except: pass
    ints.sort(key=lambda x:x["ts"])
    return uid, ints

def parse_ao(line):
    parts = line.strip().split("\t")
    if len(parts)<3: return None,[]
    uid, ints = parts[0], []
    for t in parts[2:]:
        if not t.strip(): continue
        s = t.split("|")
        if len(s)<3: continue
        try: ints.append({"dg":int(s[0]),"ts":int(s[2])})
        except: pass
    ints.sort(key=lambda x:x["ts"])
    return uid, ints

def load_data(fp, d2a, dx, dy, th, fmt):
    seqs, skip, seen = {}, 0, set()
    with open(fp,"r",encoding="utf-8") as f:
        for line in f:
            uid, ints = parse_gm(line) if fmt=="gm" else parse_ao(line)
            if uid is None or len(ints)<3: skip+=1; continue
            if fmt=="ao":
                k = tuple(i["dg"] for i in ints)
                if k in seen: continue
                seen.add(k)
            s = []
            for i in ints:
                a = d2a.get(i["dg"])
                if a is None: continue
                s.append({"item_id":a,"domain":dx if i["dg"]<th else dy,"timestamp":i["ts"]})
            if len(s)>=3: seqs[uid]=s
    logger.info(f"Loaded {fp.name}: {len(seqs)} users (skipped {skip})")
    return seqs

def build_id_map(sx, sy):
    all_i = sorted(sx|sy)
    i2i = {x:idx for idx,x in enumerate(all_i)}
    return {"item_id_to_index":i2i,"index_to_item_id":{str(v):k for k,v in i2i.items()},
            "domain_x_items":sorted(sx),"domain_y_items":sorted(sy),
            "num_items":len(all_i),"num_x_items":len(sx),"num_y_items":len(sy)}

def leave_one_out(seqs):
    tr, va, te = {},{},{}
    for uid, s in seqs.items():
        if len(s)<3: continue
        tr[uid]={"seq":s[:-2]}
        va[uid]={"seq":s[:-2],"target":s[-2]}
        te[uid]={"seq":s[:-1],"target":s[-1]}
    return tr, va, te

def convert_dataset(ds, cfg):
    fmt, ddir, dx, dy, th = cfg["fmt"], cfg["dg_data_dir"], cfg["domain_x"], cfg["domain_y"], cfg["threshold"]
    csvs = cfg["csv_paths"]
    sfx = "" if ds=="GM" else f"_{ds}"
    logger.info(f"{'='*60}\n{ds} (fmt={fmt})\n{'='*60}")

    d2a = build_mapping(csvs)
    logger.info(f"Mappings: {len(d2a)}")

    meta = build_meta(csvs, dx, dy, th)
    save_json(meta, PROCESSED_DIR/f"item_metadata{sfx}.json")

    train_s = load_data(ddir/cfg["train_file"], d2a, dx, dy, th, fmt)
    valid_s = load_data(ddir/cfg["valid_file"], d2a, dx, dy, th, fmt)
    test_s  = load_data(ddir/cfg["test_file"], d2a, dx, dy, th, fmt)

    train_d, _, _ = leave_one_out(train_s)
    valid_d = {}
    for uid,s in valid_s.items():
        if len(s)>=3: valid_d[uid]={"seq":s[:-2],"target":s[-2]}
    test_d = {}
    for uid,s in test_s.items():
        if len(s)>=3: test_d[uid]={"seq":s[:-1],"target":s[-1]}

    save_json(train_d, PROCESSED_DIR/f"train{sfx}.json")
    save_json(valid_d, PROCESSED_DIR/f"valid{sfx}.json")
    save_json(test_d, PROCESSED_DIR/f"test{sfx}.json")

    all_s = {**train_s,**valid_s,**test_s}
    ints = {uid:[i["item_id"] for i in s] for uid,s in all_s.items()}
    save_json(ints, PROCESSED_DIR/f"interactions{sfx}.json")

    sx,sy = set(),set()
    for uid,s in all_s.items():
        for i in s:
            if i["domain"]==dx: sx.add(i["item_id"])
            else: sy.add(i["item_id"])
    save_json(build_id_map(sx, sy), PROCESSED_DIR/f"id_mapping{sfx}.json")

    logger.info(f"Done {ds}: users={len(all_s)}, items={len(sx|sy)}, x={len(sx)}, y={len(sy)}, "
                f"train={len(train_d)}, valid={len(valid_d)}, test={len(test_d)}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["GM","AO"], default="GM"); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for ds in (["GM","AO"] if a.all else [a.dataset]):
        convert_dataset(ds, DATASETS[ds])

if __name__ == "__main__":
    main()

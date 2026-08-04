"""
DeepSeek V4 API 响应 JSON → item_attributes.json
用法: cd enhancement/ && python src/convert_item_prompt.py --dataset GM
"""
import argparse, csv, json, logging, re
from pathlib import Path
from typing import Dict, List

from data_utils import (PROJECT_ROOT, PROCESSED_DIR,
    ITEM_PROMPT_GM_DIR, ITEM_PROMPT_AO_DIR, DG_GM_DIR, DG_AO_DIR, save_json)

logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    "GM": {
        "csv_files": [DG_GM_DIR / "item_listM_F.csv", DG_GM_DIR / "item_listG_AM_F.csv"],
        "prompt_dir": ITEM_PROMPT_GM_DIR,
        "desc": "Game-Movie",
    },
    "AO": {
        "csv_files": [DG_AO_DIR / "item_listA_F.csv", DG_AO_DIR / "item_listO_AA_F.csv"],
        "prompt_dir": ITEM_PROMPT_AO_DIR,
        "desc": "Art-Office",
    },
}

def load_csvs(paths) -> Dict[int, str]:
    m = {}
    for p in paths:
        if not p.exists():
            logger.warning(f"CSV not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                try:
                    m[int(row[2].strip())] = row[0].strip()
                except ValueError:
                    pass
    return m

def extract(content: str) -> List[str]:
    if not content:
        return []
    try:
        a = json.loads(content)
        if isinstance(a, list):
            return [str(x).strip() for x in a if x]
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[(.*?)\]', content, re.DOTALL)
    if m:
        inner = m.group(1)
        try:
            a = json.loads(f"[{inner}]")
            if isinstance(a, list):
                return [str(x).strip() for x in a if x]
        except json.JSONDecodeError:
            pass
        return [i.strip().strip('"').strip("'") for i in inner.split(",") if i.strip()]
    return []

def convert(ds_name, csv_paths, prompt_dir):
    logger.info(f"Loading CSVs: {ds_name}")
    q2a = load_csvs(csv_paths)
    logger.info(f"  Mappings: {len(q2a)}")

    files = sorted([f for f in prompt_dir.glob("*.json") if f.name != "example_output.json"])
    logger.info(f"  Files: {len(files)}")

    attrs, total, skip_a, skip_c = {}, 0, 0, 0
    for pf in files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for e in data:
            total += 1
            q = e.get("qqid")
            if q is None:
                continue
            asin = q2a.get(q)
            if asin is None:
                skip_a += 1
                continue
            content = e.get("choices", [{}])[0].get("message", {}).get("content", "")
            al = extract(content)
            if not al:
                skip_c += 1
                continue
            attrs[asin] = {"intro": "", "attributes": al}
        logger.info(f"  {pf.name} (cumulative: {len(attrs)})")

    logger.info(f"Done {ds_name}: {len(attrs)} items, skipped asin={skip_a}, content={skip_c}")
    return attrs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["GM", "AO"], default="GM")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    for ds in (["GM", "AO"] if a.all else [a.dataset]):
        c = DATASET_CONFIG[ds]
        logger.info(f"{'='*60}\n{ds} ({c['desc']})\n{'='*60}")
        save_json(convert(ds, c["csv_files"], c["prompt_dir"]),
                  PROCESSED_DIR / f"item_attributes_{ds}.json")

if __name__ == "__main__":
    main()

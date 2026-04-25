"""GO label propagation: expand each protein's GO term set with all ancestors.

GO is a DAG: if a protein has term `GO:X`, it implicitly has every ancestor
of `GO:X` along is_a / part_of edges. Standard CAFA evaluation propagates
labels this way, which densifies the positive distribution and dramatically
raises both AUPR and Fmax for general (high-level) terms.

Outputs CSVs alongside the originals with `_propagated` suffix.
"""

import argparse
import ast
import os

import pandas as pd
import requests

OBO_URLS = [
    "https://current.geneontology.org/ontology/go-basic.obo",
    "http://purl.obolibrary.org/obo/go/go-basic.obo",
]


def download_obo(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"OBO already present at {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (MLCB-project)"}
    last_err = None
    for url in OBO_URLS:
        print(f"Downloading {url} -> {path}", flush=True)
        try:
            r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"  saved {os.path.getsize(path)/1e6:.1f} MB")
            return
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"All OBO mirrors failed: {last_err}")


def parse_obo(path: str):
    """Returns (parents, namespace, name) dicts keyed by GO id."""
    parents, namespace, name = {}, {}, {}
    cur_id, cur_parents, cur_ns, cur_name = None, set(), None, None
    obsolete, in_term = False, False

    def commit():
        if cur_id and in_term and not obsolete:
            parents[cur_id] = cur_parents
            namespace[cur_id] = cur_ns
            name[cur_id] = cur_name

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit()
                cur_id, cur_parents, cur_ns, cur_name = None, set(), None, None
                obsolete, in_term = False, True
            elif line.startswith("[") and line.endswith("]"):
                commit()
                cur_id, cur_parents, cur_ns, cur_name = None, set(), None, None
                obsolete, in_term = False, False
            elif in_term:
                if line.startswith("id: GO:"):
                    cur_id = line[4:].strip()
                elif line.startswith("name: "):
                    cur_name = line[6:].strip()
                elif line.startswith("namespace: "):
                    cur_ns = line[11:].strip()
                elif line.startswith("is_a: GO:"):
                    cur_parents.add(line.split()[1])
                elif line.startswith("relationship: part_of GO:"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "part_of":
                        cur_parents.add(parts[2])
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
        commit()

    return parents, namespace, name


def build_ancestor_map(parents: dict) -> dict:
    """Memoized DFS over the DAG -> {go_id: set of all ancestors}."""
    cache: dict = {}

    def visit(t):
        if t in cache:
            return cache[t]
        result = set()
        for p in parents.get(t, set()):
            result.add(p)
            result |= visit(p)
        cache[t] = result
        return result

    for t in list(parents.keys()):
        visit(t)
    return cache


def propagate(go_terms, ancestors):
    out = set()
    for t in go_terms:
        out.add(t)
        if t in ancestors:
            out |= ancestors[t]
    return sorted(out)


def parse_cell(cell):
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    return ast.literal_eval(cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", default="data/raw/go-basic.obo")
    ap.add_argument("--csvs", nargs="+", default=[
        "data/processed/train_dataset.csv",
        "data/processed/val_dataset.csv",
        "data/processed/test_dataset.csv",
    ])
    ap.add_argument("--out-suffix", default="_propagated")
    args = ap.parse_args()

    download_obo(args.obo)
    print("Parsing OBO...", flush=True)
    parents, namespace, _ = parse_obo(args.obo)
    print(f"  {len(parents)} non-obsolete GO terms")
    by_ns = {}
    for t, ns in namespace.items():
        by_ns[ns] = by_ns.get(ns, 0) + 1
    print(f"  by namespace: {by_ns}")

    print("Building ancestor map...", flush=True)
    ancestors = build_ancestor_map(parents)
    avg_anc = sum(len(v) for v in ancestors.values()) / max(len(ancestors), 1)
    print(f"  mean ancestors per term: {avg_anc:.1f}")

    for csv in args.csvs:
        base, ext = os.path.splitext(csv)
        out = f"{base}{args.out_suffix}{ext}"

        df = pd.read_csv(csv)
        before = df["go_terms"].apply(lambda c: len(parse_cell(c)))
        before_unique = set()
        for c in df["go_terms"]:
            before_unique |= set(parse_cell(c))

        df["go_terms"] = df["go_terms"].apply(lambda c: str(propagate(parse_cell(c), ancestors)))
        df.to_csv(out, index=False)

        after = df["go_terms"].apply(lambda c: len(parse_cell(c)))
        after_unique = set()
        for c in df["go_terms"]:
            after_unique |= set(parse_cell(c))

        print(f"\n{csv} -> {out}")
        print(f"  unique GO terms: {len(before_unique)} -> {len(after_unique)} ({len(after_unique)/max(len(before_unique),1):.1f}x)")
        print(f"  mean labels/protein: {before.mean():.1f} -> {after.mean():.1f} ({after.mean()/max(before.mean(),0.01):.1f}x)")


if __name__ == "__main__":
    main()

# train_subtype_classifier.py
import os, re, joblib, numpy as np, nibabel as nib, pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

RAW = Path(os.environ["nnUNet_raw"]) / "Dataset801_PancreasROI"
RES = Path(os.environ["nnUNet_results"]) / "Dataset801_PancreasROI"

def case_from_path(p: Path) -> str:
    n = p.name
    for suf in (".nii.gz", ".nii"):
        if n.endswith(suf):
            return n[:-len(suf)]
    return p.stem

def pick_image(img_dir: Path, case: str) -> Path | None:
    for suf in (".nii.gz", ".nii"):
        cand = img_dir / f"{case}_0000{suf}"
        if cand.exists():
            return cand
    # fallback: any *_0000.nii* that starts with case
    cands = sorted(img_dir.glob(f"{case}_0000.nii*"))
    return cands[0] if cands else None

def read_label_csv(split: str):
    csv = RAW / ("train_class_labels.csv" if split == "train" else "val_class_labels.csv")
    if csv.exists():
        df = pd.read_csv(csv)  # expects columns: case, subtype
        return {str(r["case"]): int(r["subtype"]) for _, r in df.iterrows()}
    return None  # we’ll parse subtype from filename if CSVs don’t exist

def subtype_from_case(case: str):
    m = re.match(r"quiz_(\d+)_", case)
    return int(m.group(1)) if m else None

def feats(img_p: Path, msk_p: Path):
    im = nib.load(str(img_p)); I = im.get_fdata().astype(np.float32)
    L  = nib.load(str(msk_p)).get_fdata().astype(np.uint8)
    vox = float(np.prod(im.header.get_zooms()))  # mm^3
    pan = (L > 0); les = (L == 2)
    pan_ml = pan.sum() * vox / 1000.0
    les_ml = les.sum() * vox / 1000.0
    frac = float(les.sum()) / max(1, pan.sum())
    def stats(M):
        if M.sum() < 5: return [0,0,0,0,0]
        v = I[M]
        return [float(v.mean()), float(v.std()),
                float(np.percentile(v,10)), float(np.percentile(v,50)), float(np.percentile(v,90))]
    pan_stats, les_stats = stats(pan), stats(les)
    return {
        "vol_pan_ml": pan_ml, "vol_les_ml": les_ml, "les_frac": frac,
        "pan_mean": pan_stats[0], "pan_std": pan_stats[1], "pan_p50": pan_stats[3],
        "les_mean": les_stats[0], "les_std": les_stats[1], "les_p50": les_stats[3],
    }

def build_split_df(split: str):
    imgs = RAW / ("imagesTr" if split == "train" else "imagesVal")
    labs = RAW / ("labelsTr" if split == "train" else "labelsVal")
    label_map = read_label_csv(split)  # may be None

    rows = []
    labs_list = sorted(labs.glob("*.nii*"))   # handle .nii and .nii.gz
    for lab in labs_list:
        case = case_from_path(lab)
        img  = pick_image(imgs, case)
        if img is None:
            continue
        subtype = label_map.get(case) if label_map else subtype_from_case(case)
        if subtype is None:
            continue
        f = feats(img, lab); f["case"] = case; f["subtype"] = int(subtype)
        rows.append(f)

    df = pd.DataFrame(rows)
    if df.empty or "subtype" not in df.columns:
        raise RuntimeError("Could not construct dataframe with 'subtype'. "
                           "Check filename pattern or provide train/val CSVs.")
    return df

def main():
    print("Collecting GT-based features for train/val...")
    df_tr = build_split_df("train")
    df_va = build_split_df("validation")
    print("Shapes:", df_tr.shape, df_va.shape)

    feat_cols = [c for c in df_tr.columns if c not in ("case","subtype")]
    Xtr, ytr = df_tr[feat_cols].values, df_tr["subtype"].values
    Xva, yva = df_va[feat_cols].values, df_va["subtype"].values

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=800, multi_class="multinomial", class_weight="balanced")
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    print("\nValidation report:\n", classification_report(yva, pred, digits=3))
    print("Validation macro-F1:", round(f1_score(yva, pred, average="macro"), 3))

    out_dir = RES / "cls_from_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feat_cols": feat_cols}, out_dir / "logreg_cls.joblib")
    df_tr.to_csv(out_dir / "train_features.csv", index=False)
    df_va.to_csv(out_dir / "val_features.csv", index=False)
    print("Saved artifacts to:", out_dir)

if __name__ == "__main__":
    main()

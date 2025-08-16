# predict_test_subtypes.py
import os, joblib, numpy as np, nibabel as nib, pandas as pd
from pathlib import Path

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
    cands = sorted(img_dir.glob(f"{case}_0000.nii*"))
    return cands[0] if cands else None

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

def auto_pick_test_pred_dir(res_dir: Path) -> Path:
    # prefer 3d_fullres, else 2d; if multiple, pick most recent test_pred_* directory
    candidates = [p for p in res_dir.glob("test_pred_*") if p.is_dir()]
    if not candidates:
        raise RuntimeError("No test_pred_* folder found in results. Run nnUNetv2_predict on imagesTs first.")
    # sort by mtime, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def main():
    # load model + feat cols
    model_art = RES / "cls_from_masks" / "logreg_cls.joblib"
    if not model_art.exists():
        raise RuntimeError(f"Model not found: {model_art}. Run train_subtype_classifier.py first.")
    bundle = joblib.load(model_art)
    clf, feat_cols = bundle["model"], bundle["feat_cols"]

    TESTP = auto_pick_test_pred_dir(RES)
    print("Using test predictions from:", TESTP)

    rows = []
    imgsTs = RAW / "imagesTs"
    for pred_path in sorted(TESTP.glob("*.nii*")):
        case = case_from_path(pred_path)
        img_path = pick_image(imgsTs, case)
        if img_path is None:
            continue
        f = feats(img_path, pred_path)
        X = np.array([[f[c] for c in feat_cols]])
        rows.append({"Names": case, "Subtype": int(clf.predict(X)[0])})

    out_csv = RES / "subtype_results.csv"
    pd.DataFrame(rows).sort_values("Names").to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()

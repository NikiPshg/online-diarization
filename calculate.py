from diar_utils import *
from pathlib import Path

from pyannote.core import Annotation
import pandas as pd
from pyannote.metrics.diarization import DiarizationErrorRate

import argparse
from tqdm import tqdm


def load_excel_annote(path: str):
    df = pd.read_excel(path)
    annote = Annotation()
    for _, row in df.iterrows():
        start = row['start']
        end = row['end']
        idx = row['id']
        seg = Segment(start=start, end=end)
        annote[seg] =  idx
    return annote


parser = argparse.ArgumentParser(description="Compute DER between GT and predicted RTTM/XLSX files.")
parser.add_argument('--gt_rttm', type=str, default=None, help="Path to ground-truth RTTM directory.")
parser.add_argument('--pred_rttm', type=str, default=None, help="Path to prediction RTTM directory.")
parser.add_argument('--gt_xlsx', type=str, default=None, help="Path to ground-truth XLSX directory.")
parser.add_argument('--pred_xlsx', type=str, default=None, help="Path to prediction XLSX directory.")
args = parser.parse_args()

# Determine file types and paths
gt_is_xlsx = args.gt_xlsx is not None
pred_is_xlsx = args.pred_xlsx is not None

print(f"DEBUG: gt_is_xlsx={gt_is_xlsx}, pred_is_xlsx={pred_is_xlsx}")
print(f"DEBUG: args.gt_xlsx={args.gt_xlsx}, args.pred_xlsx={args.pred_xlsx}")
print(f"DEBUG: args.gt_rttm={args.gt_rttm}, args.pred_rttm={args.pred_rttm}")

if gt_is_xlsx:
    path_to_gt = args.gt_xlsx
    gt_files = list(Path(path_to_gt).glob("*.xlsx"))
    print(f"DEBUG: Looking for XLSX files in {path_to_gt}, found {len(gt_files)} files")
else:
    if args.gt_rttm is None:
        raise ValueError("Either --gt_rttm or --gt_xlsx must be provided")
    path_to_gt = args.gt_rttm
    gt_files = list(Path(path_to_gt).glob("*.rttm"))
    print(f"DEBUG: Looking for RTTM files in {path_to_gt}, found {len(gt_files)} files")

if pred_is_xlsx:
    path_to_pred = args.pred_xlsx
    pred_files = list(Path(path_to_pred).glob("*.xlsx"))
    print(f"DEBUG: Looking for XLSX files in {path_to_pred}, found {len(pred_files)} files")
else:
    if args.pred_rttm is None:
        raise ValueError("Either --pred_rttm or --pred_xlsx must be provided")
    path_to_pred = args.pred_rttm
    pred_files = list(Path(path_to_pred).glob("*.rttm"))
    print(f"DEBUG: Looking for RTTM files in {path_to_pred}, found {len(pred_files)} files")



gt_dict = {f.stem: f for f in gt_files}
pred_dict = {f.stem: f for f in pred_files}

print(f"DEBUG: gt_dict has {len(gt_dict)} entries, pred_dict has {len(pred_dict)} entries")
print(f"DEBUG: gt_dict keys: {list(gt_dict.keys())}")
print(f"DEBUG: pred_dict keys: {list(pred_dict.keys())}")

matched_gts = []
matched_preds = []

for stem in gt_dict:
    print(f"DEBUG: Processing stem={stem}")
    if stem in pred_dict:
        print(f"DEBUG: Found match for stem={stem}")
        # Load GT annotation
        print(f"DEBUG: gt_is_xlsx={gt_is_xlsx}, loading GT annotation...")
        if gt_is_xlsx:
            print(f"DEBUG: Loading XLSX GT file: {str(gt_dict[stem])}")
            gt_annote = load_excel_annote(str(gt_dict[stem]))
            print(f"DEBUG: Loaded GT annotation: {gt_annote}")
        else:
            print(f"DEBUG: Loading RTTM GT file: {str(gt_dict[stem])}")
            gt_annote = load_rttm_annote(gt_dict[stem])
            print(f"DEBUG: Loaded GT annotation: {gt_annote}")
        
        # Load prediction annotation
        print(f"DEBUG: pred_is_xlsx={pred_is_xlsx}, loading prediction annotation...")
        if pred_is_xlsx:
            print(f"DEBUG: Loading XLSX prediction file: {str(pred_dict[stem])}")
            pred_annote = load_excel_annote(str(pred_dict[stem]))
            print(f"DEBUG: Loaded prediction annotation: {pred_annote}")
        else:
            print(f"DEBUG: Loading RTTM prediction file: {str(pred_dict[stem])}")
            pred_annote = load_rttm_annote(pred_dict[stem])
            print(f"DEBUG: Loaded prediction annotation: {pred_annote}")
        
        matched_gts.append(gt_annote)
        matched_preds.append(pred_annote)
    else:
        print(f"DEBUG: No match found for stem={stem} in pred_dict")



der = DiarizationErrorRate()

for gt_annote, pred_annote in tqdm(zip(matched_gts, matched_preds)):
    der(gt_annote, pred_annote, uri='test')

der.report(display=True)
print()
    
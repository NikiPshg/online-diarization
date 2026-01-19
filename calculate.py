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


parser = argparse.ArgumentParser(description="Compute DER between GT and predicted RTTM files.")
parser.add_argument('--gt_rttm', type=str, required=True, help="Path to ground-truth RTTM directory.")
parser.add_argument('--pred_rttm', type=str, required=True, help="Path to prediction RTTM directory.")
args = parser.parse_args()

path_to_gt_rttm = args.gt_rttm
path_to_pred_rttm = args.pred_rttm

gt_rttm_files = list(Path(path_to_gt_rttm).glob("*.rttm"))
pred_rttm_files = list(Path(path_to_pred_rttm).glob("*.rttm"))


gt_dict = {f.stem: f for f in gt_rttm_files}
pred_dict = {f.stem: f for f in pred_rttm_files}

matched_gts = []
matched_preds = []

for stem in gt_dict:
    if stem in pred_dict:
        gt_annote = load_rttm_annote(gt_dict[stem])
        pred_annote = load_rttm_annote(pred_dict[stem])
        matched_gts.append(gt_annote)
        matched_preds.append(pred_annote)



der = DiarizationErrorRate()

for gt_annote, pred_annote in tqdm(zip(matched_gts, matched_preds)):
    der(gt_annote, pred_annote, uri='test')

der.report(display=True)
print()
    
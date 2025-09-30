import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm

from models import (
    ThreeSigmaModel,
    KDEModel,
    OCSVMModel,
    EnsembleModel,
    BaseAnomalyModel,
)

BASE = '/Users/yiyu/Desktop/Output/Pyleaf/Anoamly_detection_algorithm1/KPI-Anomaly-Detection-master/Finals_dataset'
DEFAULT_RAW = os.path.join(BASE, 'phase2_ground_truth.csv')
DEFAULT_WIN = os.path.join(BASE, 'processed_ground_truth.csv')

WIN_H = 180
WIN_P = 10


def load_raw_per_kpi(raw_csv: str):
    dtype_map = {'timestamp': np.int64, 'value': np.float64, 'label': np.int8, 'KPI ID': 'category'}
    usecols = ['timestamp', 'value', 'label', 'KPI ID']
    df = pd.read_csv(raw_csv, usecols=usecols, dtype=dtype_map)
    per = {}
    for kpi_id, sub in df.groupby('KPI ID'):
        sub = sub.sort_values('timestamp').reset_index(drop=True)
        per[kpi_id] = sub
    return per


def build_model(name: str, args: argparse.Namespace) -> BaseAnomalyModel:
    name = name.lower()
    if name == '3sigma' or name == 'sigma':
        return ThreeSigmaModel(k=args.k)
    if name == 'kde':
        return KDEModel(bandwidth=args.bandwidth, kernel=args.kernel, q=args.q)
    if name == 'ocsvm':
        return OCSVMModel(nu=args.nu, gamma=args.gamma, kernel='rbf')
    if name == 'ensemble':
        # 默认3个基模型，阈值多数决
        models: List[BaseAnomalyModel] = [
            ThreeSigmaModel(k=args.k),
            KDEModel(bandwidth=args.bandwidth, kernel=args.kernel, q=args.q),
            OCSVMModel(nu=args.nu, gamma=args.gamma, kernel='rbf'),
        ]
        return EnsembleModel(models=models, threshold=args.threshold)
    raise ValueError(f'unknown model name: {name}')


def infer_points(win_df: pd.DataFrame, raw_per_kpi: dict, model: BaseAnomalyModel, max_windows: int | None = None, show_progress: bool = True):
    y_true_points: List[int] = []
    y_pred_points: List[int] = []

    it = win_df.iterrows()
    if show_progress:
        it = tqdm(it, total=len(win_df), desc='Running windows')
    for idx, row in it:
        if max_windows is not None and idx >= max_windows:
            break
        kpi = row['kpi_id']
        h0, h1 = int(row['hist_start_ts']), int(row['hist_end_ts'])
        p0, p1 = int(row['pred_start_ts']), int(row['pred_end_ts'])

        sub = raw_per_kpi.get(kpi)
        if sub is None:
            continue
        H = sub[(sub['timestamp'] >= h0) & (sub['timestamp'] <= h1)]
        P = sub[(sub['timestamp'] >= p0) & (sub['timestamp'] <= p1)]
        if len(H) != WIN_H or len(P) != WIN_P:
            continue

        flags = model.fit(H['value'].values).predict(P['value'].values)
        y_pred_points.extend(flags.astype(int).tolist())
        y_true_points.extend(P['label'].astype(int).tolist())

    return np.array(y_true_points, dtype=int), np.array(y_pred_points, dtype=int)


def main():
    parser = argparse.ArgumentParser(description='Run anomaly models on processed windows (point-level evaluation)')
    parser.add_argument('--raw', default=DEFAULT_RAW, help='path to phase2_ground_truth.csv')
    parser.add_argument('--win', default=DEFAULT_WIN, help='path to processed_ground_truth.csv')
    parser.add_argument('--model', default='ensemble', choices=['3sigma','sigma','kde','ocsvm','ensemble'])
    parser.add_argument('--max_windows', type=int, default=None)
    parser.add_argument('--no_progress', action='store_true', help='disable progress bar')
    # 3sigma
    parser.add_argument('--k', type=float, default=3.0, help='k for 3-sigma')
    # KDE
    parser.add_argument('--bandwidth', type=float, default=0.2)
    parser.add_argument('--kernel', type=str, default='gaussian')
    parser.add_argument('--q', type=float, default=0.005, help='quantile threshold based on H-scores')
    # OCSVM
    parser.add_argument('--nu', type=float, default=0.01)
    parser.add_argument('--gamma', type=str, default='scale')
    # Ensemble
    parser.add_argument('--threshold', type=int, default=None, help='votes threshold (default=majority)')

    args = parser.parse_args()

    print('RAW:', args.raw)
    print('WIN:', args.win)
    print('MODEL:', args.model)

    win_df = pd.read_csv(args.win)
    print('windows:', len(win_df))

    raw_per_kpi = load_raw_per_kpi(args.raw)
    print('kpis:', len(raw_per_kpi))

    model = build_model(args.model, args)
    y_true, y_pred = infer_points(win_df, raw_per_kpi, model, max_windows=args.max_windows, show_progress=(not args.no_progress))

    print('points collected:', len(y_true))
    if len(y_true) == 0:
        print('No points collected. Please check inputs.')
        sys.exit(1)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print('Point-level Precision/Recall/F1 (weighted):', p, r, f1)
    print('\nclassification_report (weighted overall view):')
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == '__main__':
    main()

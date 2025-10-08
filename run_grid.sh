#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
MAIN="/Users/yiyu/Desktop/Output/Pyleaf/Anoamly_detection_algorithm1/main_copy.py"
LOGDIR="/Users/yiyu/Desktop/Output/Pyleaf/Anoamly_detection_algorithm1/logs"
mkdir -p "$LOGDIR"

echo "[INFO] Logs at $LOGDIR"

# 3sigma grid with lag
for lag in 3 5 10; do
for k in 2.5 3.0 3.5; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOGDIR/three_sigma_k${k}_lag${lag}_$ts.log"
  echo "[RUN] 3sigma k=$k lag=$lag -> $log"
  $PYTHON "$MAIN" --model 3sigma --k $k --lag_window $lag --no_progress | tee "$log"
done
done

# KDE grid with lag
for lag in 3 5 10; do
for bw in 0.1 0.2 0.3; do
  for q in 0.010 0.005 0.002; do
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/kde_bw${bw}_q${q}_lag${lag}_$ts.log"
    echo "[RUN] KDE bw=$bw q=$q lag=$lag -> $log"
    $PYTHON "$MAIN" --model kde --bandwidth $bw --q $q --lag_window $lag --no_progress | tee "$log"
  done
done
done

# OCSVM grid with lag
for lag in 3 5 10; do
for nu in 0.005 0.01 0.02; do
  for gamma in scale auto; do
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/ocsvm_nu${nu}_gamma${gamma}_lag${lag}_$ts.log"
    echo "[RUN] OCSVM nu=$nu gamma=$gamma lag=$lag -> $log"
    $PYTHON "$MAIN" --model ocsvm --nu $nu --gamma $gamma --lag_window $lag --no_progress | tee "$log"
  done
done
done

# Ensemble (默认多数决) with lag
for lag in 3 5 10; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOGDIR/ensemble_default_lag${lag}_$ts.log"
  echo "[RUN] Ensemble default lag=$lag -> $log"
  $PYTHON "$MAIN" --model ensemble --lag_window $lag --no_progress | tee "$log"
done


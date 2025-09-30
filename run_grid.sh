#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
MAIN="/Users/yiyu/Desktop/Output/Pyleaf/Anoamly_detection_algorithm1/main.py"
LOGDIR="/Users/yiyu/Desktop/Output/Pyleaf/Anoamly_detection_algorithm1/logs"
mkdir -p "$LOGDIR"

echo "[INFO] Logs at $LOGDIR"

# 3sigma grid
for k in 2.5 3.0 3.5; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOGDIR/three_sigma_k${k}_$ts.log"
  echo "[RUN] 3sigma k=$k -> $log"
  $PYTHON "$MAIN" --model 3sigma --k $k --no_progress | tee "$log"
done

# KDE grid
for bw in 0.1 0.2 0.3; do
  for q in 0.010 0.005 0.002; do
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/kde_bw${bw}_q${q}_$ts.log"
    echo "[RUN] KDE bw=$bw q=$q -> $log"
    $PYTHON "$MAIN" --model kde --bandwidth $bw --q $q --no_progress | tee "$log"
  done
done

# OCSVM grid
for nu in 0.005 0.01 0.02; do
  for gamma in scale auto; do
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/ocsvm_nu${nu}_gamma${gamma}_$ts.log"
    echo "[RUN] OCSVM nu=$nu gamma=$gamma -> $log"
    $PYTHON "$MAIN" --model ocsvm --nu $nu --gamma $gamma --no_progress | tee "$log"
  done
done

# Ensemble (默认多数决)
ts=$(date +%Y%m%d_%H%M%S)
log="$LOGDIR/ensemble_default_$ts.log"
echo "[RUN] Ensemble default -> $log"
$PYTHON "$MAIN" --model ensemble --no_progress | tee "$log"


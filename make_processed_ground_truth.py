import csv
import random
from collections import defaultdict, deque
from pathlib import Path

# 配置
SRC = Path('KPI-Anomaly-Detection-master/Finals_dataset/phase2_ground_truth.csv')
OUT = Path('KPI-Anomaly-Detection-master/Finals_dataset/processed_ground_truth.csv')
WIN_H = 180   # 历史点数
WIN_P = 10    # 预测点数
WIN_LEN = WIN_H + WIN_P
FREQ_SEC = 60 # 只保留60s间隔的连续窗口
random.seed(42)

# 读取原始数据按KPI聚合
kpi_data = defaultdict(list)  # kpi_id -> list[(timestamp:int, value:str, label:int)]
with SRC.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            ts = int(r['timestamp'])
        except Exception:
            continue
        kpi = r['KPI ID']
        val = r['value']
        lb = 1 if r['label'].strip() == '1' else 0
        kpi_data[kpi].append((ts, val, lb))

# 滑窗生成（仅保留严格60s等差的连续片段）
windows_by_kpi_pos = defaultdict(list)  # 正样本窗口
windows_by_kpi_neg = defaultdict(list)  # 负样本窗口

def flush_segment(kpi_id, seg):
    if len(seg) < WIN_LEN:
        return
    dq_ts = deque()
    dq_lb = deque()
    for ts, _v, lb in seg:
        dq_ts.append(ts)
        dq_lb.append(lb)
        if len(dq_ts) > WIN_LEN:
            dq_ts.popleft()
            dq_lb.popleft()
        if len(dq_ts) == WIN_LEN:
            pred_lbs = list(dq_lb)[WIN_H:]
            y = 1 if any(pred_lbs) else 0
            hist_start_ts = dq_ts[0]
            hist_end_ts   = dq_ts[WIN_H - 1]
            pred_start_ts = dq_ts[WIN_H]
            pred_end_ts   = dq_ts[-1]
            window_id = f"{kpi_id}:{hist_start_ts}:{pred_end_ts}"
            rec = (kpi_id, hist_start_ts, hist_end_ts, pred_start_ts, pred_end_ts, y, window_id)
            if y == 1:
                windows_by_kpi_pos[kpi_id].append(rec)
            else:
                windows_by_kpi_neg[kpi_id].append(rec)

for kpi, arr in kpi_data.items():
    if not arr:
        continue
    arr.sort(key=lambda x: x[0])
    segment = []
    prev_ts = None
    for ts, v, lb in arr:
        if prev_ts is None or ts - prev_ts == FREQ_SEC:
            segment.append((ts, v, lb))
        else:
            flush_segment(kpi, segment)
            segment = [(ts, v, lb)]
        prev_ts = ts
    flush_segment(kpi, segment)

# 按每KPI做 1:3 采样（阳性全保留，阴性随机下采样到3倍阳性）
all_windows = []
total_pos = total_neg = 0
all_kpis = sorted(set(windows_by_kpi_pos.keys()) | set(windows_by_kpi_neg.keys()))
for kpi in all_kpis:
    pos_list = windows_by_kpi_pos.get(kpi, [])
    neg_list = windows_by_kpi_neg.get(kpi, [])
    n_pos = len(pos_list)
    if n_pos == 0:
        continue  # 无阳性则跳过该KPI
    keep_neg = min(len(neg_list), 3 * n_pos)
    neg_sample = random.sample(neg_list, keep_neg) if keep_neg > 0 else []
    all_windows.extend(pos_list)
    all_windows.extend(neg_sample)
    total_pos += n_pos
    total_neg += len(neg_sample)

# 写出 processed_ground_truth.csv（窗口索引）
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8', newline='') as fw:
    writer = csv.writer(fw)
    writer.writerow(['kpi_id','hist_start_ts','hist_end_ts','pred_start_ts','pred_end_ts','window_label','window_id','freq_sec'])
    for (kpi, h0, h1, p0, p1, y, wid) in all_windows:
        writer.writerow([kpi, h0, h1, p0, p1, y, wid, FREQ_SEC])

print('[DONE] 写出:', str(OUT))
print('窗口总数:', len(all_windows))
print('阳性窗口数:', total_pos)
print('阴性窗口数(采样后):', total_neg)
print('涉及KPI数:', len(set(w[0] for w in all_windows)))

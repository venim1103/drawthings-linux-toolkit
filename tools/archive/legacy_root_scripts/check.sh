#!/bin/bash
cd /workspaces/drawthings-linux-toolkit && pid=$1 && in_file='dt-models/ltx_2.3_22b_distilled_f16.ckpt' && out_file='dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix_20260602.ckpt' && if ! ps -p "$pid" >/dev/null 2>&1; then echo "status=not-running"; out_size=$(stat -c%s "$out_file" 2>/dev/null || echo 0); ./.venv/bin/python - <<'PY'
import os
out_size=float(os.popen("stat -c%s dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix_20260602.ckpt 2>/dev/null || echo 0").read().strip() or 0)
print(f"out_gb={out_size/1024/1024/1024:.2f}")
PY
exit 0; fi && read_b=$(awk '/read_bytes/ {print $2}' "/proc/$pid/io") && etime=$(ps -o etimes= -p "$pid" | tr -d ' ') && in_size=$(stat -c%s "$in_file") && out_size=$(stat -c%s "$out_file" 2>/dev/null || echo 0) && export READ_B="$read_b" ETIME="$etime" IN_SIZE="$in_size" OUT_SIZE="$out_size" && ./.venv/bin/python - <<'PY'
import os, math
rb=float(os.environ['READ_B']); et=max(1.0,float(os.environ['ETIME'])); ins=float(os.environ['IN_SIZE']); outs=float(os.environ['OUT_SIZE'])
pct=min(100.0, rb/ins*100.0 if ins else 0.0)
rate=rb/et
rem=max(0.0, ins-rb)
eta=rem/rate if rate>0 else float('inf')
h=int(eta//3600) if math.isfinite(eta) else 0
m=int((eta%3600)//60) if math.isfinite(eta) else 0
s=int(eta%60) if math.isfinite(eta) else 0
eta_hms=(f"{h:02d}:{m:02d}:{s:02d}" if math.isfinite(eta) else "unknown")
print(f"status=running")
print(f"progress_pct={pct:.2f}")
print(f"eta_hms={eta_hms}")
print(f"read_gb={rb/1024/1024/1024:.2f}")
print(f"input_gb={ins/1024/1024/1024:.2f}")
print(f"out_gb={outs/1024/1024/1024:.2f}")
print(f"rate_mbps_avg={rate/1024/1024:.2f}")
PY
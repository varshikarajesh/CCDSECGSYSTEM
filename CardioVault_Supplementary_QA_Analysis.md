# CardioVault SD Express Evaluation — Supplementary QA & Telemetry Analysis

**Status**: Supplementary to the submitted SDA competition report. Not part of the official submission. Retained in the repository for reviewers who want deeper telemetry evidence behind the report's headline claims.

**Relationship to submitted report**: The competition report (`CardioVault_SDA_Report_SRMIST.docx`, submitted Sept 16, 2026) stands on its own and is unmodified by this document. Nothing here contradicts or supersedes any claim in that report. This document exists to give reviewers deeper telemetry access if they want it.

---

## Purpose

This supplementary evaluation adds system telemetry to the existing CardioVault SD Express benchmark. Three tools were evaluated, all run on the NVIDIA Jetson Orin Nano Developer Kit at 15 W power mode, SD Express device exposed as `/dev/nvme0n1` (filesystem at `/dev/nvme0n1p1`):

1. **jtop** — Jetson SoC temperature, power, and fan telemetry
2. **ioping** — Filesystem-level storage latency
3. **iostat** — System-level device utilization and I/O statistics

---

## 1. jtop Telemetry

### Test configuration
120-second direct sequential-write fio workload, sampled ~once per second:
```
filename: /mnt/sd_express/fio_testfile
rw: write | block size: 1M | numjobs: 1
runtime: 120s | time_based: 1 | direct: 1 | size: 4G
```
Collected via the Python `jtop` API (the CLI client returned `Error connection`, though the service and API were functional).

### Results (120 samples, 16:00:23–16:02:22)

| Metric | Minimum | Average | Maximum |
|---|---:|---:|---:|
| CPU temperature | 47.06°C | 47.99°C | 48.56°C |
| GPU temperature | 47.44°C | 48.29°C | 48.91°C |
| SoC temperature | 47.53°C | 48.29°C | 48.78°C |
| Junction temperature | 47.56°C | 48.36°C | 48.91°C |
| Total input power | 3559 mW | 4598 mW | 5000 mW |
| CPU/GPU/CV power | 525 mW | 739 mW | 885 mW |
| SoC power | 1132 mW | 1386 mW | 1477 mW |
| Fan PWM duty | 24.71% | 25.50% | 27.84% |

Peak SoC 48.78°C (16:02:18); peak GPU/junction 48.91°C (16:02:15); peak power 5000 mW (16:00:41).

### Interpretation
This rerun remained thermally stable and **did not reproduce** the original SMART-reported 86–87°C event / `critical_warning 0x2` disclosed in the submitted report's Appendix A. This is not evidence against the original event — it occurred under different run conditions (duration, job count, total data volume) that were not exactly replicated here. **The original thermal disclosure in the submitted report is authoritative and unchanged.** This jtop data is an independent, non-reproducing supplementary data point, not a correction.

---

## 2. ioping Latency Profiling

### 2.1 Idle baseline
`sudo ioping -c 20 -i 0.5 /` (ext4, `/dev/nvme0n1p1`)

| Metric | Value |
|---|---:|
| Requests completed | 19 / 20 |
| Average latency | 553.9 µs |
| Min / Max latency | 282.4 µs / 974.7 µs |
| Mean deviation | 234.4 µs |
| IOPS | 1.80k |

(First request excluded as warm-up by `ioping` itself.)

### 2.2 Initial post-workload measurement (discarded as primary evidence)
A preliminary run measured latency *after* fio had already finished, not concurrently. Average 897.6 µs, min 293.8 µs, max 1.27 ms. **Not used as the primary result** since it doesn't reflect true under-load conditions — retained here only for completeness.

### 2.3 Concurrent ioping + fio (valid under-load test)
`ioping` started first; fio launched 2 seconds later.
```
ioping: sudo ioping -c 240 -i 0.5 /
fio: same 120s sequential-write config as above
```

| ioping metric | Value |
|---|---:|
| Requests completed | 239 / 240 |
| Average latency | 604.0 µs |
| Min / Max latency | 147.7 µs / 5.11 ms |
| Mean deviation | 658.2 µs |

| Concurrent fio metric | Value |
|---|---:|
| Average throughput | 600 MiB/s |
| Total data written | 70.3 GiB |
| Average IOPS | 599.92 |
| Device utilization | 100.00% |

### Interpretation
Average filesystem latency rose from 553.9 µs (idle) to 604.0 µs (under 100%-utilized sustained write) — an increase of ~9%. This is a modest, well-behaved latency increase given full device saturation. Max latency (5.11 ms) shows occasional outliers, but the average stayed well under 1 ms.

**Relevance to the submitted report**: the report's Section 3.4 concurrency test found request throughput degrades at high concurrency and attributed this to compute/GPU contention rather than storage. This ioping result independently supports that attribution — storage latency stays broadly stable under saturation, so the concurrency slowdown is not originating at the storage layer.

---

## 3. iostat System-Level Monitoring

### Test configuration
`iostat -x 1`, run concurrently with the same 120s sequential-write fio workload.

### fio results (this run)

| Metric | Value |
|---|---:|
| Average throughput | 616 MiB/s (646 MB/s) |
| Total data written | 72.2 GiB |
| Average IOPS | 616.60 |
| Fio-reported utilization | 99.99% |
| Avg completion / total latency | 1.53 ms / 1.62 ms |

Latency percentiles: p50 1.47 ms, p90 1.52 ms, p95 1.57 ms, p99 5.60 ms, p99.9 6.26 ms.

### iostat results (`nvme0n1`)

| Metric | Value |
|---|---:|
| Write rate | ~637,952–638,000 kB/s |
| Write await | ~1.22 ms |
| Avg queue size | ~2.27 |
| Device utilization | ~99.60–100.40% |
| CPU iowait | 13.21% |
| CPU idle | 80.60% |

### Interpretation
Confirms the fio workload generated genuine sustained active I/O (not an idle/lightly-loaded interval). Utilization and throughput figures corroborate fio's own reporting. Does not add hardware temperature/power data (see jtop, above) — it is a system-level cross-check, not a new independent finding.

---

## 4. Consolidated Summary

| Test | Main result | What it adds |
|---|---|---|
| jtop | Peak SoC 48.78°C, peak power 5.00 W | Stable thermal/power behavior in this rerun; did not reproduce the original 86–87°C event (report's disclosure stands unchanged) |
| ioping (idle) | 553.9 µs avg latency | Baseline |
| ioping (under load) | 604.0 µs avg, 5.11 ms max | ~9% latency increase under 100% utilization — supports "concurrency slowdown is compute-bound, not storage-bound" |
| iostat | ~100% utilization, ~638 MB/s | System-level confirmation of sustained active I/O during the test |
| *(reference)* Original SMART event | 86–87°C, `critical_warning 0x2` | Unchanged — remains the report's authoritative thermal disclosure |

## 5. Evidence Files

```
/home/vishal/jtop_seqrw_log.jsonl
/home/vishal/sda_eval/fio/sustained_write_jtop.json
/home/vishal/sda_eval/fio/sustained_write_jtop_summary.txt
/home/vishal/ioping_idle_sdexpress.txt
/home/vishal/ioping_seqwrite_sdexpress.txt
/home/vishal/ioping_during_seqwrite_sdexpress.txt
/home/vishal/ioping_seqwrite_summary.txt
/home/vishal/iostat_seqwrite_sdexpress.txt
```

## 6. Conclusion

This supplementary telemetry adds useful depth to the submitted evaluation but is intentionally kept separate from it. Nothing here overturns a claim in the submitted report: the jtop non-reproduction is reported honestly as non-reproduction, not as a correction, and the ioping/iostat results corroborate (rather than introduce) the report's existing compute-bound interpretation of the concurrency test. Reviewers wanting the fuller telemetry picture behind the submitted numbers can find it here; the submitted report remains the authoritative source for the competition record.

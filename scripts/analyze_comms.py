"""Analyze communication kernels in a Kineto trace.

Reports per-kernel and aggregated bandwidth (algBw / busBw, nccl-tests convention),
classifies each NCCL/RCCL kernel as exposed vs overlapped with compute, and breaks
down by collective type and message-size bucket.

Usage:
    python scripts/analyze_comms.py --trace-json results/.../trace_step52.json
    python scripts/analyze_comms.py --trace-json trace.json --peak-bw 128 --link-count 7

Bandwidth definitions (nccl-tests convention):
  - algBw = sendBytes / kernel_time           (perceived per-rank bandwidth)
  - busBw = algBw * scaling_factor             (comparable to peak bus BW)
      AllReduce:                       factor = 2*(N-1)/N
      AllGather / ReduceScatter:       factor = (N-1)/N
      AllToAll / AllToAllV:            factor = (N-1)/N
      Broadcast / Reduce / SendRecv:   factor = 1.0

Exposed vs overlapped:
  - "Compute kernels" = all GPU kernels EXCEPT NCCL.
  - For each NCCL kernel, the time-overlap with the union of compute intervals is
    "overlapped"; the rest is "exposed" (on the critical path).
"""
import argparse
import json
from collections import defaultdict


# Bytes per element for NCCL datatypes encountered in PyTorch profiler args.
DTYPE_BYTES = {
    'Char': 1, 'Byte': 1, 'Bool': 1,
    'Short': 2, 'Half': 2, 'BFloat16': 2,
    'Int': 4, 'Float': 4,
    'Long': 8, 'Double': 8,
    # NCCL raw enum names (some traces emit these)
    'ncclInt8': 1, 'ncclUint8': 1,
    'ncclInt32': 4, 'ncclUint32': 4,
    'ncclInt64': 8, 'ncclUint64': 8,
    'ncclFloat16': 2, 'ncclBfloat16': 2,
    'ncclFloat32': 4, 'ncclFloat64': 8,
}


def dtype_bytes(dt, default=4):
    if dt is None:
        return default
    return DTYPE_BYTES.get(str(dt), default)


def bus_factor(coll, n):
    """nccl-tests bus-bandwidth scaling factor for a given collective and rank count."""
    if n <= 1:
        return 1.0
    if coll in ('allreduce', 'all_reduce'):
        return 2.0 * (n - 1) / n
    if coll in ('allgather', 'all_gather', 'reduce_scatter', 'reducescatter'):
        return (n - 1) / n
    if coll in ('all_to_all', 'all_to_allv', 'alltoall', 'alltoallv'):
        return (n - 1) / n
    return 1.0


def union(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_us(a_s, a_e, intervals, idx_hint=0):
    """Sum of overlap between [a_s, a_e] and union list 'intervals'.

    Returns (overlap, new_idx_hint). Caller passes intervals sorted by start
    and reuses idx_hint for sequential scans.
    """
    ov = 0.0
    i = idx_hint
    while i < len(intervals) and intervals[i][1] < a_s:
        i += 1
    j = i
    while j < len(intervals) and intervals[j][0] < a_e:
        ov += max(0.0, min(a_e, intervals[j][1]) - max(a_s, intervals[j][0]))
        j += 1
    return ov, i


def classify_collective(coll_name):
    """Normalize collective name to a canonical short tag."""
    if coll_name is None:
        return 'unknown'
    nl = coll_name.lower()
    if 'all_to_all' in nl or 'alltoall' in nl:
        return 'all_to_all'
    if 'all_reduce' in nl or 'allreduce' in nl:
        return 'all_reduce'
    if 'all_gather' in nl or 'allgather' in nl:
        return 'all_gather'
    if 'reduce_scatter' in nl or 'reducescatter' in nl:
        return 'reduce_scatter'
    if 'broadcast' in nl:
        return 'broadcast'
    if 'reduce' in nl:
        return 'reduce'
    return nl


def size_bucket(bytes_val):
    if bytes_val < 1 << 10:
        return '<1KB'
    if bytes_val < 1 << 20:
        return '<1MB'
    if bytes_val < 1 << 23:
        return '<8MB'
    if bytes_val < 1 << 26:
        return '<64MB'
    if bytes_val < 1 << 30:
        return '<1GB'
    return '>=1GB'


def infer_purpose(coll, dtype, n_in, n_out, name):
    """Heuristic purpose label based on (collective, dtype, msg-size, kernel name).

    These patterns match TorchRec EmbeddingShardingDist phases on a typical
    OneTrans / DLRM training step. Override / extend as new patterns appear.
    """
    is_msccl = 'mscclKernel' in name
    if coll == 'all_to_all':
        if dtype == 'Long' and n_in <= 32 and n_out <= 32:
            return 'KJT splits a2a (per-feature lengths metadata)'
        if dtype == 'Int' and 1_000 <= n_in < 1_000_000 and n_in == n_out:
            return 'KJT keys a2a (sparse feature indices)'
        if dtype == 'Long' and 1_000_000 <= max(n_in, n_out) < 100_000_000:
            return 'KJT lengths a2a (per-row offsets)'
        if dtype == 'Float' and max(n_in, n_out) >= 100_000_000:
            # FWD: small in (per-rank batch shard) → big out (full features)
            # BWD: big in (full grads) → small out (per-rank batch shard)
            return 'Embedding data a2a (FWD dispatch / BWD gradient)'
        return f'all_to_all ({dtype}, {max(n_in, n_out)} elems)'
    if coll == 'all_reduce':
        if dtype == 'Float':
            return 'DDP gradient AllReduce (dense parameter sync)'
        return f'all_reduce ({dtype})'
    if coll == 'all_gather':
        return 'all_gather'
    if coll == 'reduce_scatter':
        return 'reduce_scatter'
    return f'{coll} ({dtype})'


def collect(events):
    """Walk events once. Return (nccl_list, compute_intervals, n_steps, step_us, first_step_ts).

    nccl_list rows: dict with start, end, dur_us, coll, group_size,
                    bytes_per_rank (max of in/out × dtype_bytes), dtype, name.
    step_us = average ProfilerStep wall duration (used as % denominator).
    first_step_ts = earliest ts of any ProfilerStep event (or None).
    """
    nccl = []
    compute = []
    step_durs_by_name = {}
    first_step_ts = None

    for e in events:
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat')
        if cat == 'kernel':
            n = e.get('name', '')
            ts = float(e['ts'])
            dur = float(e.get('dur', 0))
            if dur <= 0:
                continue
            if 'ncclDev' in n or 'mscclKernel' in n:
                a = e.get('args', {}) or {}
                coll = classify_collective(a.get('Collective name'))
                grp = a.get('Group size') or 0
                try:
                    grp = int(grp)
                except (ValueError, TypeError):
                    grp = 0
                in_nel = a.get('In msg nelems', 0) or 0
                out_nel = a.get('Out msg nelems', 0) or 0
                try:
                    nel = max(int(in_nel), int(out_nel))
                except (ValueError, TypeError):
                    nel = 0
                dt = a.get('dtype')
                bpr = nel * dtype_bytes(dt)
                nccl.append({
                    'start': ts, 'end': ts + dur, 'dur_us': dur,
                    'coll': coll, 'n_ranks': grp,
                    'bytes_per_rank': bpr, 'dtype': str(dt) if dt else 'None',
                    'name': n.split('<')[0],
                    'in_nel': int(in_nel) if isinstance(in_nel, (int, float)) else 0,
                    'out_nel': int(out_nel) if isinstance(out_nel, (int, float)) else 0,
                    'grid': tuple(a.get('grid', ())) if a.get('grid') else None,
                    'block': tuple(a.get('block', ())) if a.get('block') else None,
                })
            else:
                compute.append((ts, ts + dur))
        nm = e.get('name', '')
        if 'ProfilerStep' in nm and e.get('ph') == 'X' and e.get('dur', 0) > 0:
            ts_e = float(e['ts'])
            if first_step_ts is None or ts_e < first_step_ts:
                first_step_ts = ts_e
            # Dedupe by step name (multiple threads emit the same span)
            step_durs_by_name[nm] = max(step_durs_by_name.get(nm, 0), float(e['dur']))

    n_steps = len(step_durs_by_name) if step_durs_by_name else 1
    step_us = (sum(step_durs_by_name.values()) / n_steps) if step_durs_by_name else 0
    nccl.sort(key=lambda x: x['start'])
    return nccl, compute, n_steps, step_us, first_step_ts


def report(nccl, compute, n_steps, step_us, args):
    cu = union(compute)
    idx = 0
    for k in nccl:
        ov, idx = overlap_us(k['start'], k['end'], cu, idx)
        k['exposed_us'] = max(0.0, k['dur_us'] - ov)
        k['overlapped_us'] = ov
        n = k['n_ranks'] or args.world_size
        k['n_ranks_eff'] = n
        k['algBw_GBs'] = (k['bytes_per_rank'] / k['dur_us'] * 1e-3) if k['dur_us'] > 0 else 0
        k['busBw_GBs'] = k['algBw_GBs'] * bus_factor(k['coll'], n)
        k['purpose'] = infer_purpose(k['coll'], k['dtype'], k['in_nel'], k['out_nel'], k['name'])

    total_us = sum(k['dur_us'] for k in nccl)
    exp_us = sum(k['exposed_us'] for k in nccl)
    ov_us = sum(k['overlapped_us'] for k in nccl)
    total_bytes = sum(k['bytes_per_rank'] for k in nccl)

    print(f"=" * 80)
    print(f"COMMUNICATION ANALYSIS")
    print(f"=" * 80)
    print(f"  trace          : {args.trace_json}")
    print(f"  profiled steps : {n_steps}")
    print(f"  step wall time : {step_us/1000:.1f} ms (avg across {n_steps} ProfilerStep events)")
    print(f"  NCCL kernels   : {len(nccl)}")
    print(f"  total NCCL time: {total_us/1000:.1f} ms ({total_us/n_steps/1000:.1f} ms/step, "
          f"{total_us/(step_us*n_steps)*100:.1f}% of step)" if step_us else "")
    print(f"  total per-rank wire bytes: {total_bytes/1e9:.2f} GB ({total_bytes/n_steps/1e9:.3f} GB/step)")
    print(f"  exposed (critical path)  : {exp_us/1000:.1f} ms "
          f"({exp_us/n_steps/1000:.1f} ms/step, "
          f"{exp_us/(step_us*n_steps)*100:.1f}% of step)" if step_us else "")
    print(f"  overlapped with compute  : {ov_us/1000:.1f} ms "
          f"({ov_us/n_steps/1000:.1f} ms/step, "
          f"{ov_us/(step_us*n_steps)*100:.1f}% of step)" if step_us else "")

    if args.peak_bw > 0:
        link_peak = args.peak_bw * args.link_count
        print(f"\n  reference per-rank XGMI/NVLink peak: {args.peak_bw:.0f} GB/s × {args.link_count} links = {link_peak:.0f} GB/s")

    link_peak = args.peak_bw * args.link_count if args.peak_bw > 0 else 0

    def fmt_bw(gbs):
        if gbs >= 1.0:
            s = f"{gbs:6.1f} GB/s"
        elif gbs >= 0.001:
            s = f"{gbs*1000:6.1f} MB/s"
        else:
            s = f"{gbs*1e6:6.1f} KB/s"
        if link_peak > 0:
            s += f" ({gbs/link_peak*100:4.1f}%)"
        return s

    def fmt_size(b):
        if b >= 1e9:
            return f"{b/1e9:.2f} GB"
        if b >= 1e6:
            return f"{b/1e6:.1f} MB"
        if b >= 1e3:
            return f"{b/1e3:.1f} KB"
        return f"{b} B"

    # ===== Unified table: each kernel group sorted by EXPOSED time first =====
    # Group by (purpose, dtype, msg-size-bucket) so that 5 events of the same
    # collective collapse into one row (avg msg/dur, sum total time, etc).
    # Sort key: (-exposed_us, -total_us) so most critical-path-impactful kinds
    # sit at the top, fully-hidden ones drop to the bottom (sub-sorted by total).
    print(f"\n{'='*135}")
    print(f"PER-COMM-KIND BREAKDOWN — SORTED BY EXPOSED TIME (critical path) FIRST, then total time")
    print(f"  Where = how this kind's time decomposes: EXPOSED on critical path vs HIDDEN behind compute")
    print(f"{'='*135}")
    groups = defaultdict(lambda: {
        'n': 0, 'tot_dur': 0.0, 'tot_exp': 0.0, 'tot_bytes': 0,
        'msg_per_call': 0, 'busBw_w': 0.0, 'algBw_w': 0.0,
        'variants_count': defaultdict(int),  # (kernel_short, grid, block) → count
        'variants_set': set(),
        'rep_kernel': None,  # filled after first pass
    })
    for k in nccl:
        key = (k['purpose'], k['dtype'], size_bucket(k['bytes_per_rank']))
        g = groups[key]
        g['n'] += 1
        g['tot_dur'] += k['dur_us']
        g['tot_exp'] += k['exposed_us']
        g['tot_bytes'] += k['bytes_per_rank']
        g['busBw_w'] += k['busBw_GBs'] * k['bytes_per_rank']
        g['algBw_w'] += k['algBw_GBs'] * k['bytes_per_rank']
        # msg sizes can vary within a bucket (e.g., DDP grad AR has many bucket
        # sizes from 1MB to 39MB). Track min/max to detect heterogeneous groups.
        g['min_msg'] = min(g.get('min_msg', float('inf')), k['bytes_per_rank'])
        g['max_msg'] = max(g.get('max_msg', 0), k['bytes_per_rank'])
        # vol/call accumulator: per-rank I/O volume per call = (in + out) × dtype_bytes
        bw = dtype_bytes(k['dtype']) if k['dtype'] != 'None' else 0
        g['tot_io_bytes'] = g.get('tot_io_bytes', 0) + (k['in_nel'] + k['out_nel']) * bw
        variant_key = (k['name'], k['grid'], k['block'])
        g['variants_count'][variant_key] += 1
        # Human-readable variant string for listing
        nm_short = (
            'mscclKernel_*' if 'mscclKernel' in k['name']
            else 'ncclDevKernel_Generic_1' if 'ncclDevKernel_Generic_1' in k['name']
            else k['name'].split('(')[0]
        )
        g['variants_set'].add(f"{nm_short} grid={k['grid']} block={k['block']}")

    # Pick representative kernel per group = most-frequent variant
    for g in groups.values():
        most = max(g['variants_count'].items(), key=lambda x: x[1])[0]
        g['rep_kernel'] = {'name': most[0], 'grid': most[1], 'block': most[2]}

    # Width: ~210 cols. Inline: kernel/grid/block + dtype + vol/call + ms/call + algBw + busBw + timing.
    # vol/call = (In + Out) × dtype_bytes per rank — total per-call I/O volume per rank.
    # algBw = msg_size / time (perceived per-rank BW, sendBytes / time)
    # busBw = algBw × bus_factor (nccl-tests bus BW, comparable to per-link peak)
    print(f"{'#':>3s} {'purpose':<44s} {'kernel':<22s} {'grid':<13s} {'block':<11s} "
          f"{'dtype':<8s} {'#/step':>6s} {'vol/call':>12s} {'ms/call':>9s} "
          f"{'algBw':>20s} {'busBw':>20s} "
          f"{'tot ms':>7s} {'%step':>6s} {'EXP ms':>7s} {'%exp':>6s} {'where':<14s}")
    print('-' * 215)

    # Sort: exposed time desc (primary), total time desc (secondary).
    # This puts critical-path comm at the top; hidden-behind-compute kinds at
    # the bottom in order of GPU-stream occupancy.
    sorted_groups = sorted(groups.items(),
                           key=lambda x: (-x[1]['tot_exp'], -x[1]['tot_dur']))
    for i, ((purpose, dtype, bucket), g) in enumerate(sorted_groups, 1):
        avg_dur_per_step = g['tot_dur'] / n_steps
        exp_per_step = g['tot_exp'] / n_steps
        pct_step = g['tot_dur'] / (step_us * n_steps) * 100 if step_us else 0
        pct_exp = g['tot_exp'] / (step_us * n_steps) * 100 if step_us else 0
        bus_avg = g['busBw_w'] / g['tot_bytes'] if g['tot_bytes'] > 0 else 0
        alg_avg = g['algBw_w'] / g['tot_bytes'] if g['tot_bytes'] > 0 else 0
        share_exp = g['tot_exp'] / g['tot_dur'] if g['tot_dur'] > 0 else 0
        if share_exp > 0.5:
            where = f'EXPOSED ({share_exp*100:.0f}%)'
        elif share_exp < 0.05:
            where = 'HIDDEN'
        else:
            where = f'mixed ({share_exp*100:.0f}%)'
        n_per_step = g['n'] / n_steps
        avg_vol = g['tot_io_bytes'] / g['n'] if g['n'] > 0 else 0
        vol_str = fmt_size(avg_vol)
        avg_ms_per_call = g['tot_dur'] / g['n'] / 1000 if g['n'] > 0 else 0
        # Pick representative kernel from this group (most common)
        rep = g['rep_kernel']
        rep_name_short = (
            'mscclKernel_Sum' if 'mscclKernel' in rep['name']
            else 'ncclDevKernel_G1' if 'ncclDevKernel_Generic_1' in rep['name']
            else rep['name'].split('(')[0][:21]
        )
        grid_str = str(rep['grid']) if rep['grid'] else 'n/a'
        block_str = str(rep['block']) if rep['block'] else 'n/a'
        # dtype: append byte width so the user can audit msg/call computation
        dtype_str = dtype if dtype != 'None' else '?'
        bw = dtype_bytes(dtype) if dtype != 'None' else 0
        dtype_label = f'{dtype_str}({bw}B)' if bw else dtype_str
        print(f"{i:>3d} {purpose[:43]:<44s} {rep_name_short:<22s} {grid_str:<13s} {block_str:<11s} "
              f"{dtype_label:<8s} {n_per_step:>5.1f}× {vol_str:>12s} {avg_ms_per_call:>8.2f}m "
              f"{fmt_bw(alg_avg):>20s} {fmt_bw(bus_avg):>20s} "
              f"{avg_dur_per_step/1000:>6.2f} {pct_step:>5.1f}% "
              f"{exp_per_step/1000:>6.2f} {pct_exp:>5.1f}% {where:<14s}")

    print('-' * 215)
    tot_per_step = total_us / n_steps
    exp_per_step = exp_us / n_steps
    print(f"{'':>3s} {'TOTAL':<44s} {'':>22s} {'':>13s} {'':>11s} "
          f"{'':<8s} {len(nccl)/n_steps:>5.1f}× {'':>12s} {'':>9s} "
          f"{'':>20s} {'':>20s} "
          f"{tot_per_step/1000:>6.2f} {tot_per_step/step_us*100:>5.1f}% "
          f"{exp_per_step/1000:>6.2f} {exp_per_step/step_us*100:>5.1f}%  EXP+HID")

    # If a group has multiple distinct (grid, block) variants, list them
    multi_variant = [(i, names_set) for i, ((p, _, _), g) in enumerate(sorted_groups, 1)
                     if len((variants := g['variants_set'])) > 1
                     for names_set in [variants]]
    if multi_variant:
        print(f"\nGroups with multiple kernel variants (showed dominant in main table):")
        for idx, variants in multi_variant:
            for v in sorted(variants):
                print(f"  #{idx}: {v}")

    # ===== By collective type =====
    print(f"\n{'='*100}")
    print(f"BREAKDOWN BY COLLECTIVE TYPE (per step; busBw vs per-rank XGMI/NVLink peak)")
    print(f"{'='*100}")
    print(f"{'collective':<14s} {'kerns':>5s} {'total ms':>10s} {'exp ms':>8s} {'hidden %':>9s} "
          f"{'total GB':>9s} {'avg algBw':>20s} {'avg busBw':>20s}")
    print('-' * 100)
    by_coll = defaultdict(lambda: {'n':0,'tot':0.0,'exp':0.0,'bytes':0,'algBw':0.0,'busBw':0.0,'wbytes':0.0})
    for k in nccl:
        c = by_coll[k['coll']]
        c['n'] += 1
        c['tot'] += k['dur_us']
        c['exp'] += k['exposed_us']
        c['bytes'] += k['bytes_per_rank']
        # weighted avg by bytes (bigger transfers dominate avg BW)
        c['algBw'] += k['algBw_GBs'] * k['bytes_per_rank']
        c['busBw'] += k['busBw_GBs'] * k['bytes_per_rank']
        c['wbytes'] += k['bytes_per_rank']
    for coll, d in sorted(by_coll.items(), key=lambda x: -x[1]['tot']):
        hidden_pct = (d['tot'] - d['exp']) / d['tot'] * 100 if d['tot'] > 0 else 0
        alg = d['algBw'] / d['wbytes'] if d['wbytes'] > 0 else 0
        bus = d['busBw'] / d['wbytes'] if d['wbytes'] > 0 else 0
        print(f"{coll:<14s} {d['n']:>5d} {d['tot']/n_steps/1000:>9.1f} {d['exp']/n_steps/1000:>7.1f} "
              f"{hidden_pct:>8.1f}% {d['bytes']/n_steps/1e9:>8.2f} "
              f" {fmt_bw(alg):>18s} {fmt_bw(bus):>18s}")

    # ===== By size bucket =====
    print(f"\n{'='*100}")
    print(f"BREAKDOWN BY MESSAGE SIZE BUCKET × COLLECTIVE (per step)")
    print(f"{'='*100}")
    print(f"{'bucket':<10s} {'collective':<14s} {'kerns':>5s} {'total ms':>10s} {'exp ms':>8s} "
          f"{'hidden %':>9s} {'avg busBw':>20s}")
    print('-' * 90)
    by_bucket = defaultdict(lambda: {'n':0,'tot':0.0,'exp':0.0,'busBw':0.0,'wbytes':0.0})
    for k in nccl:
        bk = (size_bucket(k['bytes_per_rank']), k['coll'])
        c = by_bucket[bk]
        c['n'] += 1
        c['tot'] += k['dur_us']
        c['exp'] += k['exposed_us']
        c['busBw'] += k['busBw_GBs'] * k['bytes_per_rank']
        c['wbytes'] += k['bytes_per_rank']
    # sort by descending total time
    for bk, d in sorted(by_bucket.items(), key=lambda x: -x[1]['tot']):
        hidden_pct = (d['tot'] - d['exp']) / d['tot'] * 100 if d['tot'] > 0 else 0
        bus = d['busBw'] / d['wbytes'] if d['wbytes'] > 0 else 0
        print(f"{bk[0]:<10s} {bk[1]:<14s} {d['n']:>5d} {d['tot']/n_steps/1000:>9.1f} "
              f"{d['exp']/n_steps/1000:>7.1f} {hidden_pct:>8.1f}% {fmt_bw(bus):>20s}")

    # ===== Top exposed kernels =====
    print(f"\n{'='*110}")
    print(f"TOP 15 KERNELS BY EXPOSED TIME (per-step contribution to critical path)")
    print(f"{'='*110}")
    print(f"{'rank':<5s} {'collective':<11s} {'bytes/rank':>11s} {'dur us':>9s} "
          f"{'exposed us':>11s} {'busBw':>20s} {'dtype':<6s} {'grid':<14s}")
    print('-' * 105)
    rows = sorted(nccl, key=lambda x: -x['exposed_us'])
    for i, k in enumerate(rows[:15]):
        sz = k['bytes_per_rank']
        sz_s = f"{sz/1e9:.2f}GB" if sz >= 1e9 else (f"{sz/1e6:.0f}MB" if sz >= 1e6 else (
            f"{sz/1e3:.0f}KB" if sz >= 1000 else f"{sz}B"))
        print(f"{i+1:<5d} {k['coll']:<11s} {sz_s:>11s} {k['dur_us']:>8.0f}us "
              f"{k['exposed_us']:>9.0f}us  {fmt_bw(k['busBw_GBs']):>18s} "
              f"{k['dtype']:<6s} {str(k['grid']):<14s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace-json', required=True)
    ap.add_argument('--world-size', type=int, default=8,
                    help='Default n_ranks if not present in kernel args (default 8)')
    ap.add_argument('--peak-bw', type=float, default=0,
                    help='Per-link peak BW in GB/s (e.g. 128 for XGMI4, 50 for NVLink5)')
    ap.add_argument('--link-count', type=int, default=7,
                    help='Number of P2P links per GPU in fully-meshed topology (8 GPUs → 7)')
    ap.add_argument('--drop-unknown-args', action='store_true',
                    help='Drop NCCL kernels with no Collective name / grid args (trace-boundary '
                         'artifacts where the CPU launch op was emitted before profiler '
                         'attachment)')
    args = ap.parse_args()

    with open(args.trace_json) as f:
        d = json.load(f)
    events = d['traceEvents'] if isinstance(d, dict) else d

    nccl, compute, n_steps, step_us, first_step_ts = collect(events)
    if not nccl:
        print('No NCCL kernels found in trace.')
        return

    # Filter NCCL kernels to those that start after the first ProfilerStep begins.
    # Avoids double-counting trace-boundary artifacts (kernels left over from a
    # step that wasn't profiled, or with stripped args because their CPU launch
    # op preceded profiler attachment).
    if first_step_ts is not None:
        before = len(nccl)
        dropped_us = sum(k['dur_us'] for k in nccl if k['start'] < first_step_ts)
        nccl = [k for k in nccl if k['start'] >= first_step_ts]
        n_dropped = before - len(nccl)
        if n_dropped > 0:
            print(f"[filter] Dropped {n_dropped} NCCL kernels (total {dropped_us/1000:.1f} ms) "
                  f"that started before the first ProfilerStep at ts={first_step_ts:.0f}.")

    if args.drop_unknown_args:
        # Heuristic: drop NCCL kernels with no Collective name (typically trace-boundary
        # artifacts where args were stripped). Only sensible if you have a
        # well-instrumented trace.
        before = len(nccl)
        dropped_us = sum(k['dur_us'] for k in nccl
                         if k['name'].startswith('ncclDevKernel') and k['bytes_per_rank'] == 0
                            and k['grid'] is None)
        nccl = [k for k in nccl
                if not (k['name'].startswith('ncclDevKernel') and k['bytes_per_rank'] == 0
                        and k['grid'] is None)]
        n_dropped = before - len(nccl)
        if n_dropped > 0:
            print(f"[filter] --drop-unknown-args removed {n_dropped} NCCL kernels with no "
                  f"Collective name / grid / msg-size args (total {dropped_us/1000:.1f} ms).")

    report(nccl, compute, n_steps, step_us, args)


if __name__ == '__main__':
    main()

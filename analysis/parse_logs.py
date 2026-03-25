import re
import os
import json
from pathlib import Path
from collections import defaultdict


DATETIME_PAT = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\]\s+')

EPOCH_PAT     = re.compile(r'Epoch (\d+) loss: ([\d.]+); train accuracy: ([\d.]+); test accuracy: ([\d.]+)')
ROUND_START   = re.compile(r'Round (\d+) starts')
CLIENT_START  = re.compile(r'Client (\d+) Starts')
LAMBDA_STATE  = re.compile(r'Client (\d+) Post-Training State -> Raw \u03bb:\s*([\d.]+) \| s\(p\): ([\d.]+) \| Truly Effective \u03bb \(for W\):\s*([\d.]+)')
ADAPT_LAMBDA  = re.compile(r'Data entropy: ([\d.]+), Normalized: ([\d.]+) -> Adaptive Lambda: ([\d.]+)')
CLIENT_INIT   = re.compile(r'Client (\d+): Calculated initial lambda = ([\d.]+)')
G_PROTO_STD   = re.compile(r'g_protos_std \(global internal\): ([\d.]+)')
INTER_STD     = re.compile(r'inter_client_proto_std \(cross-client\): ([\d.]+)')
FINAL_ACC     = re.compile(r'The test accuracy of (\S+): ([\d.]+)')
MODEL_VAR     = re.compile(r'Model variance: mean: ([\d.]+), sum: ([\d.]+)')


def strip_ts(line):
    return DATETIME_PAT.sub('', line).strip()


def parse_log(log_path):
    records = dict(
        path=str(log_path),
        rounds=[],
        init_lambdas=[],
        method_accs=defaultdict(list),
    )

    current_round = None
    current_client = None
    round_obj = None

    with open(log_path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = strip_ts(raw)

            m = ROUND_START.search(line)
            if m:
                current_round = int(m.group(1))
                round_obj = dict(
                    round=current_round,
                    clients={},
                    g_proto_std=None,
                    inter_client_std=None,
                    model_var_mean=None,
                    final_accs={},
                )
                records['rounds'].append(round_obj)
                continue

            m = CLIENT_START.search(line)
            if m:
                current_client = int(m.group(1))
                if round_obj is not None and current_client not in round_obj['clients']:
                    round_obj['clients'][current_client] = dict(
                        epochs=[],
                        raw_lambda=None,
                        s_p=None,
                        eff_lambda=None,
                    )
                continue

            m = EPOCH_PAT.search(line)
            if m and round_obj is not None and current_client is not None:
                client_obj = round_obj['clients'].setdefault(current_client, dict(
                    epochs=[], raw_lambda=None, s_p=None, eff_lambda=None
                ))
                client_obj['epochs'].append(dict(
                    epoch=int(m.group(1)),
                    loss=float(m.group(2)),
                    train_acc=float(m.group(3)),
                    test_acc=float(m.group(4)),
                ))
                continue

            m = LAMBDA_STATE.search(line)
            if m and round_obj is not None:
                cid = int(m.group(1))
                client_obj = round_obj['clients'].setdefault(cid, dict(
                    epochs=[], raw_lambda=None, s_p=None, eff_lambda=None
                ))
                client_obj['raw_lambda']  = float(m.group(2))
                client_obj['s_p']         = float(m.group(3))
                client_obj['eff_lambda']  = float(m.group(4))
                continue

            m = CLIENT_INIT.search(line)
            if m:
                records['init_lambdas'].append(dict(
                    client=int(m.group(1)),
                    init_lambda=float(m.group(2)),
                ))
                continue

            m = ADAPT_LAMBDA.search(line)
            if m:
                records.setdefault('entropy_events', []).append(dict(
                    entropy=float(m.group(1)),
                    normalized=float(m.group(2)),
                    adaptive_lambda=float(m.group(3)),
                ))
                continue

            m = G_PROTO_STD.search(line)
            if m and round_obj is not None:
                round_obj['g_proto_std'] = float(m.group(1))
                continue

            m = INTER_STD.search(line)
            if m and round_obj is not None:
                round_obj['inter_client_std'] = float(m.group(1))
                continue

            m = MODEL_VAR.search(line)
            if m and round_obj is not None:
                round_obj['model_var_mean'] = float(m.group(1))
                continue

            m = FINAL_ACC.search(line)
            if m and round_obj is not None:
                round_obj['final_accs'][m.group(1)] = float(m.group(2))
                records['method_accs'][m.group(1)].append(float(m.group(2)))
                continue

    return records


def parse_all(log_dir):
    log_dir = Path(log_dir)
    results = {}
    for p in sorted(log_dir.rglob('*.log')):
        try:
            results[p.name] = parse_log(p)
        except Exception as e:
            print(f'WARN: could not parse {p}: {e}')
    return results


if __name__ == '__main__':
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'logs/multi_seed'
    all_results = parse_all(target)
    out = Path('analysis/parsed_logs.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'Parsed {len(all_results)} log files -> {out}')
    for name, rec in all_results.items():
        for method, accs in rec['method_accs'].items():
            if accs:
                print(f'  {name}: {method} final_acc={accs[-1]:.4f} (best={max(accs):.4f})')

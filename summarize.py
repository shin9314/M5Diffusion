"""Summarize recorded evidence only; never launches a GPU workload."""
import argparse
import json
import statistics
from pathlib import Path

STAGES = ['prepare_time', 'text_encoder_time', 'diffusion_time', 'vae_time', 'readback_time', 'latent_init_time', 'postprocess_time', 'diagnostic_time', 'stage_sum_time', 'total_time', 'wall_time', 'seconds_per_step', 'png_time']
FAIRNESS = ['request', 'model', 'sampler', 'schedule', 'rng', 'precision']

def summarize_file(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data.get('runs'):
        return None
    result = {'file': path.name, 'protocol': {k: data.get(k) for k in FAIRNESS}, 'model_load_time': data.get('model_load_time'), 'backend': data['runs'][0].get('backend'), 'quantize': data['runs'][0].get('quantize', 0), 'timing_schema': data.get('timing_schema', 'legacy-stage-sum-v1')}
    for cold, label in [(True, 'cold'), (False, 'warm')]:
        rows = [r for r in data['runs'] if bool(r.get('cold')) == cold]
        result[label] = {'n': len(rows), 'stages': {}}
        for stage in STAGES:
            values = [r[stage] for r in rows if isinstance(r.get(stage), (float, int))]
            if values:
                result[label]['stages'][stage] = {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'n': len(values)}
    result['environment'] = {k: data.get(k) for k in ['before', 'after']}
    return result

def build_report(directory, baseline=None):
    files = sorted(directory.glob('*.json'))
    runs = [r for p in files if (r := summarize_file(p)) is not None]
    comparisons = []
    base = next((r for r in runs if r['file'] == baseline), None)
    if baseline and base is None:
        raise ValueError('baseline must name an existing benchmark JSON with runs')
    if base:
        for run in runs:
            if run is base:
                continue
            mismatches = [k for k in FAIRNESS if base['protocol'][k] != run['protocol'][k]]
            if base['quantize'] != run['quantize']:
                mismatches.append('quantization')
            row = {'baseline': base['file'], 'candidate': run['file'], 'protocol_mismatches': mismatches, 'eligible_for_matched_speedup': not mismatches, 'timing_schema_match': base['timing_schema'] == run['timing_schema']}
            for stage in ['total_time', 'diffusion_time', 'wall_time']:
                a, b = base['warm']['stages'].get(stage), run['warm']['stages'].get(stage)
                comparable_timing = stage == 'diffusion_time' or row['timing_schema_match']
                if a and b and not mismatches and comparable_timing:
                    row[stage + '_warm_speedup'] = a['median'] / b['median']
            row['confidence'] = 'Descriptive only; no confidence interval. Repeated runs share one process; independent sessions and varied execution order are needed.'
            row['warm_sample_counts'] = [base['warm']['n'], run['warm']['n']]
            comparisons.append(row)
    quality = []
    for path in files:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and 'ssim' in data and 'psnr' in data:
            quality.append({'file': path.name, **data})
    return {'runs': runs, 'comparisons': comparisons, 'quality': quality, 'caveats': [
        'Timing schema v2 total_time is full generation wall time minus measured .npy diagnostics; wall_time includes diagnostics; stage_sum_time is the component sum. Cross-schema total/wall speedups are suppressed, diffusion remains comparable.',
        'cold means first generation after model load, not a cleared OS cache or cold machine.',
        'total_time in legacy results is the sum of named stage timers; it excludes model load, PNG, latent initialization, intermediate saves and postprocessing. Use wall_time where present for generation latency.',
        'A single warm sample is not a stable estimate. Medians here describe only available samples.',
        'The MPS comparator uses the same weights and custom pipeline, not end-to-end AUTOMATIC1111 UI timing.',
        'buffer_reuse stores immutable schedule metadata; physical activation allocation/reuse belongs to the MLX allocator.',
        'SSIM and PSNR are pixel similarity evidence for the compared images, not human preference or dataset-wide quality. Quantized variants require separate quality comparisons.',
        'RSS and backend allocation are overlapping views of unified memory; do not sum them. Thermal, swap and other activity can confound sequential runs.',
        'Matching model paths is not a weight checksum. Preserve model provenance and checkpoints separately.'
    ]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'benchmark/results')
    parser.add_argument('--baseline', help='Exact filename of baseline benchmark JSON')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = build_report(args.results, args.baseline)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)

if __name__ == '__main__':
    main()

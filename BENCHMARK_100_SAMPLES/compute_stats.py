import json, numpy as np

with open('results/benchmark_100_metrics.json', 'r') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')

methods = ['baseline', 'cosine', 'adaptive_tanh', 'slr_cap']
method_names = {
    'baseline': 'Baseline (Hard)',
    'cosine': 'Cosine Ablation',
    'adaptive_tanh': 'Adaptive TAGS (tanh)',
    'slr_cap': 'SLR Cap (rho=1.5)'
}

stats = {m: {'peak': [], 'mean': [], 'w': [], 'accepted': [], 'withheld': []} for m in methods}
shock_ratios = []
rms_disps = []
norm_teachers = []
norm_deltas = []
wins = {m: 0 for m in methods}

for s in data:
    shock_ratios.append(s['ratio_shock'])
    rms_disps.append(s['rms_displacement'])
    norm_teachers.append(s['norm_teacher'])
    norm_deltas.append(s['norm_delta'])
    evals = s['evaluations']

    best_m = min(methods, key=lambda m: evals[m]['peak_flux'])
    wins[best_m] += 1

    for m in methods:
        ev = evals[m]
        stats[m]['peak'].append(ev['peak_flux'])
        stats[m]['mean'].append(ev['mean_flux'])
        stats[m]['w'].append(ev['effective_w'])
        stats[m]['accepted'].append(ev['accepted_rms'])
        stats[m]['withheld'].append(ev['withheld_rms'])

print(f"Shock ratio: mean={np.mean(shock_ratios):.4f}, std={np.std(shock_ratios):.4f}, min={np.min(shock_ratios):.4f}, max={np.max(shock_ratios):.4f}")
print(f"RMS displacement: mean={np.mean(rms_disps):.4f}, std={np.std(rms_disps):.4f}, min={np.min(rms_disps):.4f}, max={np.max(rms_disps):.4f}")
print(f"Norm teacher mean: {np.mean(norm_teachers):.2f}")
print(f"Norm delta mean:   {np.mean(norm_deltas):.2f}")
print()

for m in methods:
    pk = stats[m]['peak']
    mn = stats[m]['mean']
    w  = stats[m]['w']
    ac = stats[m]['accepted']
    wi = stats[m]['withheld']
    print(f"{method_names[m]}:")
    print(f"  Peak flux: mean={np.mean(pk):.4f}, std={np.std(pk):.4f}, median={np.median(pk):.4f}, min={np.min(pk):.4f}, max={np.max(pk):.4f}")
    print(f"  Mean flux: mean={np.mean(mn):.4f}, std={np.std(mn):.4f}")
    print(f"  Weight:    mean={np.mean(w):.4f}, std={np.std(w):.4f}, min={np.min(w):.4f}, max={np.max(w):.4f}")
    print(f"  Accepted RMS: mean={np.mean(ac):.4f}")
    print(f"  Withheld RMS: mean={np.mean(wi):.4f}")
    print(f"  Peak wins: {wins[m]}/{len(data)}")
    print()

tags_pk  = np.array(stats['adaptive_tanh']['peak'])
base_pk  = np.array(stats['baseline']['peak'])
cosine_pk = np.array(stats['cosine']['peak'])
slr_pk   = np.array(stats['slr_cap']['peak'])

diffs = base_pk - tags_pk
print(f"TAGS vs Baseline peak flux improvement: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")
print(f"TAGS beats Baseline on peak flux: {int(np.sum(diffs > 0))}/{len(data)} samples")
print(f"TAGS beats Cosine on peak flux:   {int(np.sum(cosine_pk > tags_pk))}/{len(data)} samples")

ids = [s['sample_id'] for s in data]
diffs_sorted = sorted(zip(diffs.tolist(), ids), reverse=True)
print("Top 5 samples where TAGS helped most (Baseline - TAGS peak flux):")
for d, sid in diffs_sorted[:5]:
    s = next(x for x in data if x['sample_id'] == sid)
    print(f"  Sample {sid}: ratio_shock={s['ratio_shock']:.3f}, Baseline={s['evaluations']['baseline']['peak_flux']:.1f}, TAGS={s['evaluations']['adaptive_tanh']['peak_flux']:.1f}, diff={d:.1f}")

cosine_vs_base = (cosine_pk - base_pk).tolist()
c_sorted = sorted(zip(cosine_vs_base, ids), reverse=True)
print("Top 5 samples where Cosine was worst vs Baseline:")
for d, sid in c_sorted[:5]:
    s = next(x for x in data if x['sample_id'] == sid)
    print(f"  Sample {sid}: Baseline={s['evaluations']['baseline']['peak_flux']:.1f}, Cosine={s['evaluations']['cosine']['peak_flux']:.1f}, TAGS={s['evaluations']['adaptive_tanh']['peak_flux']:.1f}, excess={d:.1f}")

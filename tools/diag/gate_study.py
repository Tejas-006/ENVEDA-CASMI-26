# Post-hoc PubChem gate on a saved sim2 run (needs loss_buckets.py output): when a rule fires, PubChem top-1 leads the final list
import json, sys
import numpy as np, pandas as pd
d = sys.argv[1]; sub_name = sys.argv[2] if len(sys.argv) > 2 else 'sub_final_A.csv'
r = pd.read_csv(d + 'buckets.csv').set_index('mid')
pc = json.load(open(d + 'pc_lists.json')); b = json.load(open(d + 'base_pre_ice.json'))
f = lambda x: x
sig = pd.DataFrame({m: dict(lib_max=float(b[m][2]) if m in b and len(b[m]) > 2 else np.nan,
                            S=pc[m].get('S'), pop=pc[m].get('top_pop'),
                            rel=(pc[m]['fz_top'] if pc[m].get('fz_top') is not None else (pc[m]['pc_fz'] or [np.nan])[0]) - pc[m]['best_pool_fz']
                                if pc[m].get('best_pool_fz') is not None and np.isfinite(pc[m]['best_pool_fz']) else np.nan,
                            n_pc=len(pc[m].get('pc', [])))
                    for m in pc}).T.astype(float)
sig.index = sig.index.astype(str); r.index = r.index.astype(str)
r = r.join(sig)
rr = lambda x: (1 / x).fillna(0)
fin, p = rr(r.final), rr(r.pc.where(r.pc <= 25))
# rank of the truth after promoting PubChem top-1: 1 if PubChem top-1 is the truth; else final rank shifted by 1 (if PubChem top-1 was above it)
def after(fire):
    new = fin.copy()
    pc1 = r.pc == 1
    new[fire & pc1] = 1.0
    shift = fire & ~pc1 & r.final.notna()
    new[shift] = 1 / (r.final[shift] + 1).clip(upper=26).where(r.final[shift] + 1 <= 25, np.inf)
    return new
print(f'n={len(r)}  base MRR {fin.mean():.4f}')
gain = (r.pc == 1) & (r.final != 1); loss = (r.final == 1) & (r.pc != 1)
print(f'gain group (PubChem top-1 right, final top-1 wrong): {gain.sum()}  loss group (final top-1 right, PubChem top-1 wrong): {loss.sum()}')
print(sig.describe().T[['mean', '25%', '50%', '75%']].round(2))
print('signal medians  gain vs loss:')
print(pd.DataFrame({'gain': r[gain][['lib_max', 'S', 'pop', 'rel']].median(), 'loss': r[loss][['lib_max', 'S', 'pop', 'rel']].median()}).round(2))
print('\nrule                              fires   MRR     delta')
for name, fire in [('always', pd.Series(True, index=r.index)),
                   ('lib_max < 0.5', r.lib_max < 0.5), ('lib_max < 0.6', r.lib_max < 0.6), ('lib_max < 0.7', r.lib_max < 0.7),
                   ('S > 6', r.S > 6), ('S > 8', r.S > 8), ('pop >= 5', r['pop'] >= 5),
                   ('rel > 0', r.rel > 0), ('rel > 100', r.rel > 100),
                   ('lib<0.6 & rel>0', (r.lib_max < 0.6) & (r.rel > 0)), ('lib<0.6 & S>6', (r.lib_max < 0.6) & (r.S > 6))]:
    a = after(fire.fillna(False))
    print(f'{name:32s} {int(fire.fillna(False).sum()):5d}  {a.mean():.4f}  {a.mean() - fin.mean():+.4f}')
r.to_csv(d + 'gate_signals.csv')

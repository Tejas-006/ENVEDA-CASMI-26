# CMatch training: spectrum encoder + molecule encoder, contrastive against same-formula decoys.
# Needs from the setup cell: find(), cmatch (imported), OUT dir. Holds out the sim2 compounds (same selection code
# and seed as tools/sim2/sim_setup.py) so sim2 can score CMatch on compounds it never trained on.
EPOCHS, BATCH, DECOYS, MAX_SPEC_PER_COMP, VAL_FRAC = 30, 512, 4, 8, 0.02
SIM_SEED, SIM_HOLDOUT, TIME_BUDGET_S = 2026, 5000, 9 * 3600
import os, time, pickle, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from multiprocessing import Pool as MPool
T0 = time.time()
log = lambda *a: print(f'[cmatch] {time.time() - T0:7.0f}s', *a, flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(0)
TR_PATH = find('train.parquet')


def _k14(k):
    if isinstance(k, (bytes, np.bytes_)):
        k = k.decode('ascii', 'ignore')
    return str(k).strip().upper()[:14]


def _inst(x):
    return str(x).lower().replace('-', '').replace(' ', '') if x is not None else 'unk'


# ---- 1. compounds, and the sim2 hold-out (identical to sim_setup.py's ordering) ----
meta = pq.read_table(TR_PATH, columns=['inchikey14', 'normalized_smiles', 'instrument_type', 'adduct', 'ionization_mode',
                                       'precursor_mz', 'molecular_formula']).to_pandas()
meta['row'] = np.arange(len(meta))
meta['inst'] = [_inst(x) for x in meta.instrument_type]
meta = meta.dropna(subset=['inchikey14', 'normalized_smiles'])
meta['inchikey14'] = [_k14(k) for k in meta.inchikey14]
NP_KEYS = set()
for _f in ['coco_meta.pkl', 'bio_meta.pkl']:
    try:
        NP_KEYS |= {_k14(k) for k in pickle.load(open(find(_f), 'rb'))['keys'] if k is not None}
    except Exception as e:
        log('NP keys not loaded', _f, repr(e))
comp = meta.groupby('inchikey14').agg(smiles=('normalized_smiles', 'first'), n=('row', 'size'),
                                      n_inst=('inst', 'nunique'), formula=('molecular_formula', 'first')).reset_index()
comp['has_tims'] = comp.inchikey14.isin(set(meta.inchikey14[meta.inst == 'timstof']))
elig = comp[comp.inchikey14.isin(NP_KEYS)]
if len(elig) < 1000:
    elig = comp
_r = np.random.default_rng(SIM_SEED)
_order = np.argsort(-(np.log(_r.random(len(elig))) / (1.0 + 2.0 * elig.has_tims.values)), kind='stable')
HOLDOUT = set(elig.inchikey14.values[_order[:SIM_HOLDOUT]])
log(f'{len(meta):,} spectra / {len(comp):,} compounds; sim2 hold-out {len(HOLDOUT):,} (first: {elig.inchikey14.values[_order[0]]})')

comp = comp[~comp.inchikey14.isin(HOLDOUT)].reset_index(drop=True)
is_val = rng.random(len(comp)) < VAL_FRAC
ci = pd.Series(np.arange(len(comp)), index=comp.inchikey14)

# ---- 2. molecule features for every kept compound ----
with MPool(os.cpu_count()) as mp:
    feats = mp.map(cmatch.mol_features, list(comp.smiles), chunksize=500)
ok = np.array([f is not None for f in feats])
MOLF = np.zeros((len(comp), cmatch.MOL_DIM), np.uint8)
MOLF[ok] = np.stack([f for f in feats if f is not None])
del feats
log(f'molecule features: {ok.sum():,} of {len(comp):,} parsed')

# isomer groups for hard negatives: same formula, else nearest by formula string neighbour
grp = comp.groupby('formula').indices
GROUPS = {f: np.asarray(ix)[ok[ix]] for f, ix in grp.items() if isinstance(f, str)}
form_of = comp.formula.values

# ---- 3. pick training spectra (<= MAX_SPEC_PER_COMP per compound) and extract top-k peaks ----
m2 = meta[meta.inchikey14.isin(ci.index)].copy()
m2['c'] = ci.reindex(m2.inchikey14).values
m2 = m2[ok[m2.c.values]]
m2 = m2.sample(frac=1.0, random_state=0).groupby('c').head(MAX_SPEC_PER_COMP).sort_values('row')
rows = m2.row.values
N = len(rows)
SMZ = np.zeros((N, cmatch.K_PEAKS), np.float32); SIT = np.zeros((N, cmatch.K_PEAKS), np.float32)
pf = pq.ParquetFile(TR_PATH)
want = np.zeros(pf.metadata.num_rows, bool); want[rows] = True
pos = np.full(pf.metadata.num_rows, -1, np.int64); pos[rows] = np.arange(N)
off = 0
for g in range(pf.num_row_groups):
    t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities'])
    n = t.num_rows
    loc = np.where(want[off:off + n])[0]
    if len(loc):
        mzc = t.column(0).combine_chunks(); itc = t.column(1).combine_chunks()
        o = mzc.offsets.to_numpy(); mv = mzc.values.to_numpy(zero_copy_only=False); iv = itc.values.to_numpy(zero_copy_only=False)
        for j in loc:
            SMZ[pos[off + j]], SIT[pos[off + j]] = cmatch.spectrum_arrays(mv[o[j]:o[j + 1]], iv[o[j]:o[j + 1]])
    off += n
    del t
SPREC = m2.precursor_mz.values.astype(np.float32)
SADD = np.array([cmatch.adduct_id(a) for a in m2.adduct.values], np.int64)
SMODE = (m2.ionization_mode.astype(str).str.lower() == 'positive').values.astype(np.int64)
STIMS = (m2.inst.values == 'timstof').astype(np.int64)
SCOMP = m2.c.values.astype(np.int64)
spec_of = pd.Series(np.arange(N)).groupby(SCOMP).apply(np.asarray).to_dict()
log(f'training spectra: {N:,} for {len(spec_of):,} compounds')

train_c = np.array([c for c in spec_of if not is_val[c]])
val_c = np.array([c for c in spec_of if is_val[c]])


def decoys(c, k):
    g = GROUPS.get(form_of[c])
    if g is not None and len(g) > 1:
        cand = g[g != c]
        return rng.choice(cand, k, replace=len(cand) < k)
    return rng.choice(train_c, k)


def spec_batch(sidx):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    return (t(SMZ[sidx]), t(SIT[sidx]), t(SPREC[sidx]), t(SADD[sidx], torch.long), t(SMODE[sidx], torch.long),
            t(STIMS[sidx], torch.long))


# ---- 4. train ----
model = cmatch.CMatch().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
steps_per_epoch = max(1, len(train_c) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=EPOCHS * steps_per_epoch, pct_start=0.05)
scaler = torch.amp.GradScaler(enabled=dev == 'cuda')


@torch.no_grad()
def evaluate(cs, max_n=2000, max_cand=100):
    model.eval()
    rr = []
    for c in cs[:max_n]:
        g = GROUPS.get(form_of[c])
        if g is None or len(g) < 2:
            continue
        cand = g if len(g) <= max_cand else np.concatenate([[c], rng.choice(g[g != c], max_cand - 1, replace=False)])
        q = model.spec(*spec_batch(spec_of[c][:1]))
        s = (model.mol(torch.as_tensor(MOLF[cand], device=dev)) @ q.T).squeeze(1).cpu().numpy()
        rr.append(1.0 / (1 + int((s > s[list(cand).index(c)]).sum())))
    model.train()
    return (float(np.mean(rr)) if rr else float('nan')), len(rr)


hist = []
stop = False
for ep in range(EPOCHS):
    perm = rng.permutation(train_c)
    tot = 0.0
    for b in range(steps_per_epoch):
        cs = perm[b * BATCH:(b + 1) * BATCH]
        sidx = np.array([rng.choice(spec_of[c]) for c in cs])
        mols = np.concatenate([cs] + [decoys(c, DECOYS) for c in cs]).astype(np.int64)
        with torch.autocast(dev, enabled=dev == 'cuda'):
            q = model.spec(*spec_batch(sidx))
            m = model.mol(torch.as_tensor(MOLF[mols], device=dev))
            logits = model.logit_scale.exp().clamp(max=100) * q @ m.T
            same = torch.as_tensor(mols[None, :] == cs[:, None], device=dev)
            same[torch.arange(len(cs)), torch.arange(len(cs))] = False   # duplicates of the positive aren't negatives
            logits = logits.masked_fill(same, -1e4)
            loss = F.cross_entropy(logits.float(), torch.arange(len(cs), device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
        if time.time() - T0 > TIME_BUDGET_S:
            stop = True
            break
    vm, vn = evaluate(val_c)
    hist.append(dict(epoch=ep + 1, loss=tot / max(1, b + 1), val_isomer_mrr=vm, val_n=vn))
    log(f'epoch {ep + 1}/{EPOCHS} loss {tot / max(1, b + 1):.4f} | val isomer MRR {vm:.4f} (n={vn})')
    if stop:
        log('time budget reached, stopping early')
        break

OUT = '/kaggle/working/cmatch_out'
os.makedirs(OUT, exist_ok=True)
model.save(os.path.join(OUT, 'cmatch.pt'), history=hist, holdout_n=len(HOLDOUT))
json.dump(dict(history=hist, n_train_spectra=N, n_compounds=len(comp), holdout=sorted(HOLDOUT)),
          open(os.path.join(OUT, 'cmatch_meta.json'), 'w'))
import shutil
shutil.copy(cmatch.__file__, os.path.join(OUT, 'cmatch.py'))
log('saved', sorted(os.listdir(OUT)))
print(json.dumps(hist[-3:], indent=1))

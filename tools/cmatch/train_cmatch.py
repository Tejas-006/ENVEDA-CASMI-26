# CMatch training: spectrum encoder + molecule encoder, contrastive against same-formula decoys.
# Needs from the setup cell: find(), cmatch (imported), OUT dir. Holds out the sim2 compounds (same selection code
# and seed as tools/sim2/sim_setup.py) so sim2 can score CMatch on compounds it never trained on.
EPOCHS, BATCH, DECOYS, MAX_SPEC_PER_COMP, VAL_FRAC = 30, 512, 4, 8, 0.02
SIM_SEED, SIM_HOLDOUT, TIME_BUDGET_S = 2026, 5000, 9 * 3600
EVAL_N, EVAL_MAX_CAND, PATIENCE = 2000, 100, 6
# Attempt 2 of the contrastive line: both variants are judged on the scaffold hold-out and the better one is saved
# as cmatch.pt. 'v1s' = the v1 recipe; 'reg' adds peak dropout, intensity jitter and more dropout against overfitting.
VARIANTS = [dict(name='v1s', drop=0.1, aug=False),
            dict(name='reg', drop=0.2, aug=True, peak_drop=0.2, int_jitter=0.15)]
import os, time, pickle, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from multiprocessing import Pool as MPool
T0 = time.time()
log = lambda *a: print(f'[cmatch] {time.time() - T0:7.0f}s', *a, flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
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
# validation = whole held-out Murcko scaffolds, so val compounds have no close relatives in training
# (a random compound split scored 0.64 here but didn't carry over to unseen molecules)
with MPool(os.cpu_count()) as mp:
    SCAF = mp.map(cmatch.scaffold_key, list(comp.smiles), chunksize=1000)
import zlib as _zlib
is_val = np.array([(_zlib.crc32(sc.encode()) % 10000) / 10000 < VAL_FRAC for sc in SCAF])
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
log(f'training spectra: {N:,} for {len(spec_of):,} compounds; scaffold-held-out val compounds: {int(is_val[list(spec_of)].sum()):,}')

train_c = np.array([c for c in spec_of if not is_val[c]])
GROUPS_TRAIN = {f: ix[~is_val[ix]] for f, ix in GROUPS.items()}   # decoys never come from held-out scaffolds
val_c = np.array([c for c in spec_of if is_val[c]])


def decoys(c, k, r):
    g = GROUPS_TRAIN.get(form_of[c])
    if g is not None and len(g) > 1:
        cand = g[g != c]
        return r.choice(cand, k, replace=len(cand) < k)
    return r.choice(train_c, k)


def augment(mz, it, v, r):
    """Peak dropout (never the base peak, column 0) and log-normal intensity jitter, renormalised to max 1."""
    mz, it = mz.copy(), it.copy()
    drop = (r.random(it.shape) < v['peak_drop']) & (it > 0)
    drop[:, 0] = False
    mz[drop], it[drop] = 0, 0
    it = np.where(it > 0, it * np.exp(r.normal(0, v['int_jitter'], it.shape)), 0).astype(np.float32)
    it /= np.maximum(it.max(1, keepdims=True), 1e-6)
    return mz, it


def spec_batch(sidx, mz=None, it=None):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    return (t(SMZ[sidx] if mz is None else mz), t(SIT[sidx] if it is None else it), t(SPREC[sidx]),
            t(SADD[sidx], torch.long), t(SMODE[sidx], torch.long), t(STIMS[sidx], torch.long))


def fixed_items(cs, seed):
    """Same-formula candidate lists drawn once, so every epoch and variant is scored on identical questions."""
    r = np.random.default_rng(seed)
    out = []
    for c in r.permutation(cs):
        g = GROUPS.get(form_of[c])
        if g is None or len(g) < 2:
            continue
        cand = g if len(g) <= EVAL_MAX_CAND else np.concatenate([[c], r.choice(g[g != c], EVAL_MAX_CAND - 1, replace=False)])
        out.append((c, cand, int(np.flatnonzero(cand == c)[0])))
        if len(out) == EVAL_N:
            break
    return out


VAL_ITEMS = fixed_items(val_c, 1)       # unseen scaffolds: the number that decides
SEEN_ITEMS = fixed_items(train_c, 2)    # training compounds: the gap to VAL shows how much is memorised
log(f'eval sets: scaffold val {len(VAL_ITEMS):,} | seen train {len(SEEN_ITEMS):,} (<= {EVAL_MAX_CAND} same-formula candidates)')


@torch.no_grad()
def evaluate(model, items):
    model.eval()
    rr = []
    for i in range(0, len(items), 256):
        chunk = items[i:i + 256]
        q = model.spec(*spec_batch(np.array([spec_of[c][0] for c, _, _ in chunk])))
        for (c, cand, pos_c), qi in zip(chunk, q):
            s = (model.mol(torch.as_tensor(MOLF[cand], device=dev)) @ qi).float().cpu().numpy()
            rr.append(1.0 / (1 + int((s > s[pos_c]).sum())))
    model.train()
    return (float(np.mean(rr)) if rr else float('nan')), len(rr)


# ---- 4. train each variant; keep its best epoch on the scaffold hold-out ----
def train_variant(v, budget_s):
    t0 = time.time()
    r = np.random.default_rng(0)
    torch.manual_seed(0)
    model = cmatch.CMatch(drop=v['drop']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_c) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=EPOCHS * steps_per_epoch, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    hist, best, stop = [], dict(val=-1.0, epoch=0, state=None), False
    for ep in range(EPOCHS):
        perm = r.permutation(train_c)
        tot = 0.0
        for b in range(steps_per_epoch):
            cs = perm[b * BATCH:(b + 1) * BATCH]
            sidx = np.array([r.choice(spec_of[c]) for c in cs])
            mz, it = augment(SMZ[sidx], SIT[sidx], v, r) if v['aug'] else (None, None)
            mols = np.concatenate([cs] + [decoys(c, DECOYS, r) for c in cs]).astype(np.int64)
            with torch.autocast(dev, enabled=dev == 'cuda'):
                q = model.spec(*spec_batch(sidx, mz, it))
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
            if time.time() - t0 > budget_s:
                stop = True
                break
        vm, vn = evaluate(model, VAL_ITEMS)
        sm = evaluate(model, SEEN_ITEMS)[0] if (ep + 1) % 5 == 0 or stop else float('nan')
        hist.append(dict(epoch=ep + 1, loss=tot / max(1, b + 1), val_isomer_mrr=vm, val_n=vn, seen_isomer_mrr=sm))
        log(f'[{v["name"]}] epoch {ep + 1}/{EPOCHS} loss {tot / max(1, b + 1):.4f} | scaffold-val isomer MRR {vm:.4f} (n={vn})'
            + (f' | seen-train {sm:.4f}' if sm == sm else ''))
        if vm > best['val']:
            best = dict(val=vm, epoch=ep + 1, state={k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
        elif ep + 1 - best['epoch'] >= PATIENCE:
            log(f'[{v["name"]}] no scaffold-val gain for {PATIENCE} epochs, stopping')
            break
        if stop:
            log(f'[{v["name"]}] time budget reached, stopping early')
            break
    model.load_state_dict(best['state'])
    seen_best = evaluate(model, SEEN_ITEMS)[0]
    log(f'[{v["name"]}] best epoch {best["epoch"]}: scaffold-val {best["val"]:.4f} | seen-train {seen_best:.4f} '
        f'| {time.time() - t0:.0f}s')
    return model, dict(name=v['name'], variant=v, best_epoch=best['epoch'], val_isomer_mrr=best['val'],
                       seen_isomer_mrr=seen_best, secs=round(time.time() - t0), history=hist)


OUT = '/kaggle/working/cmatch_out'
os.makedirs(OUT, exist_ok=True)
results, best_model = [], None
for k, v in enumerate(VARIANTS):
    left = TIME_BUDGET_S - (time.time() - T0) - 600
    model, res = train_variant(v, left / (len(VARIANTS) - k))
    model.save(os.path.join(OUT, f'cmatch_{v["name"]}.pt'), history=res['history'], holdout_n=len(HOLDOUT))
    results.append(res)
    if best_model is None or res['val_isomer_mrr'] > max(x['val_isomer_mrr'] for x in results[:-1]):
        best_model, best_res = model, res
    del model
    if dev == 'cuda':
        torch.cuda.empty_cache()

best_model.save(os.path.join(OUT, 'cmatch.pt'), history=best_res['history'], holdout_n=len(HOLDOUT), variant=best_res['name'])
summary = dict(winner=best_res['name'], gate_val_isomer_mrr=0.30, passes_gate=best_res['val_isomer_mrr'] >= 0.30,
               v1_random_split_val=0.642, variants=[{k: x[k] for k in x if k != 'history'} for x in results])
json.dump(dict(summary=summary, runs=results, n_train_spectra=N, n_compounds=len(comp), holdout=sorted(HOLDOUT)),
          open(os.path.join(OUT, 'cmatch_meta.json'), 'w'))
import shutil
shutil.copy(cmatch.__file__, os.path.join(OUT, 'cmatch.py'))
log('saved', sorted(os.listdir(OUT)))
print('CMATCH SUMMARY ' + json.dumps(summary))

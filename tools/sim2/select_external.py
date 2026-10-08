# sim2 question picker: public spectra (MassSpecGym, private dataset casmi26-massspecgym) of compounds that are NOT
# in the competition train set, so no model in the pipeline has seen them, as on the hidden test. The library is
# the real, untouched train set. The truth may or may not be in the candidate pool (recorded as in_pool).
EXT_FILE = 'msg_spectra.parquet'
_ext = pd.read_parquet(find(EXT_FILE))
_ext['inchikey14'] = [_k14(k) for k in _ext.inchikey14]
_ext['inst'] = [_inst(x) for x in _ext.instrument_type]
_ext = _ext.dropna(subset=['inchikey14', 'normalized_smiles']).reset_index(drop=True)
_ext['row'] = np.arange(len(_ext))
TRAIN_IKS = {_k14(k) for k in pq.read_table(TR_PATH, columns=['inchikey14']).column(0).to_pylist() if k is not None}
ecomp = _ext.groupby('inchikey14').agg(smiles=('normalized_smiles', 'first'), n=('row', 'size')).reset_index()
n_all = len(ecomp)
ecomp = ecomp[~ecomp.inchikey14.isin(TRAIN_IKS)].reset_index(drop=True)
NP_FILTER = f'external: {len(ecomp):,} of {n_all:,} {EXT_FILE} compounds are not in train (by inchikey14)'
print(NP_FILTER)

rng = np.random.default_rng(SEED)
order = rng.permutation(len(ecomp))
pick = ecomp.iloc[order[SHARD * SHARD_SIZE:(SHARD + 1) * SHARD_SIZE]].reset_index(drop=True)
assert len(pick), f'shard {SHARD} is past the end of the eligible external compounds'

try:
    POOL_KEYS = {_k14(k) for k in pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key}
except Exception as e:
    POOL_KEYS = set()
    print('pool keys not loaded (in_pool will be False):', repr(e))

by_ik = {k: g for k, g in _ext[_ext.inchikey14.isin(set(pick.inchikey14))].groupby('inchikey14')}
int_ids = pa.types.is_integer(TEST_SCHEMA.field('molecule_id').type)
q_rows, drop_rows, qmeta, n_taut = [], set(), [], 0
for qi, r in pick.iterrows():
    tk = metric_key(r.smiles)
    if tk in TRAIN_IKS:                     # tautomer of a train compound: the metric would count it as seen
        n_taut += 1
        continue
    g = by_ik[r.inchikey14]
    qinst = g.inst.value_counts().index[0]
    rows = g.row[g.inst == qinst].values
    if len(rows) > MAX_SPEC:
        rows = np.sort(np.random.default_rng(SEED + qi).choice(rows, MAX_SPEC, replace=False))
    mid = 920_000_000 + SHARD * 100_000 + qi if int_ids else f'ext{SHARD}_{qi:05d}'
    q_rows += [(int(x), mid) for x in rows]
    sel = g[g.row.isin(rows)]
    qmeta.append(dict(molecule_id=mid, inchikey14=r.inchikey14, smiles=r.smiles, truth_key=tk, sim_class=3,
                      scaffold=scaffold_of(r.smiles), query_inst=qinst, n_spec=len(rows),
                      adducts=';'.join(sorted(set(map(str, sel.adduct)))), ion_mode=str(sel.ionization_mode.iloc[0]),
                      precursor_mz=float(sel.precursor_mz.median()), formula=str(sel.molecular_formula.iloc[0]),
                      n_lib_spectra=0, in_pool=bool(r.inchikey14 in POOL_KEYS or tk in POOL_KEYS), shard=SHARD))
QUERIES = pd.DataFrame(qmeta)
TRUTH = {m: {k for k in (t, ik) if k} for m, t, ik in zip(QUERIES.molecule_id, QUERIES.truth_key, QUERIES.inchikey14)}
TRUTH_KEY = dict(zip(QUERIES.molecule_id, QUERIES.truth_key))
print(f'external queries: {len(QUERIES)} (dropped {n_taut} tautomers of train compounds); '
      f'truth in candidate pool: {QUERIES.in_pool.mean():.1%}')

_qt = _ext.iloc[[x for x, _ in q_rows]].reset_index(drop=True)
_qt['molecule_id'] = [m for _, m in q_rows]
if 'spectrum_id' in TEST_SCHEMA.names:
    _qt['spectrum_id'] = [f'ext{SHARD}_{i}' for i in range(len(_qt))] if not int_ids else \
        920_000_000 + SHARD * 100_000 * MAX_SPEC + np.arange(len(_qt))
for c in TEST_SCHEMA.names:
    if c not in _qt.columns:
        print('WARNING test column missing from external data, filled with None:', c); _qt[c] = None
_qt[TEST_SCHEMA.names].to_parquet(os.path.join(SM, 'test.parquet'), index=False)
_sc = pd.read_csv(os.path.join(REAL_COMP, 'sample_submission.csv'), nrows=1).columns
pd.DataFrame({c: (QUERIES.molecule_id.values if c == 'molecule_id' else 'CCO') for c in _sc}).to_csv(
    os.path.join(SM, 'sample_submission.csv'), index=False)
if not os.path.exists(os.path.join(SM, 'train.parquet')):
    os.symlink(TR_PATH, os.path.join(SM, 'train.parquet'))   # library untouched
QUERIES.to_parquet(os.path.join(SIM_OUT, 'queries.parquet'), index=False)
os.makedirs('work', exist_ok=True)
STRUCT_FILTER = 'off (external queries are not in train)'

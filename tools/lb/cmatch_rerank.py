# v5b (LB track): CMatch within-formula re-rank of the ranker lists, before ICEBERG/GLACIER.
# score = z(ranker) + CM_LAM * z(CMatch cosine) inside each same-formula group; groups keep their slots.
# Needs the private dataset casmi26-cmatch (cmatch.pt + cmatch.py). Any failure keeps the 0.401 behaviour.
CM_LAM = 1.0
cm_stats = dict(molecules=0, changed_top1=0, changed_top25=0)
try:
    _cm_dir = os.path.dirname(find('cmatch.pt'))
    sys.path.insert(0, _cm_dir)
    import cmatch as _cm
    _cm_model, _ = _cm.CMatch.load(os.path.join(_cm_dir, 'cmatch.pt'), dev)
    _has_inst = 'instrument_type' in te.columns

    def _z(v):
        v = np.asarray(v, np.float64)
        ok = np.isfinite(v)
        out = np.zeros(len(v))
        if ok.sum() > 1 and v[ok].std() > 0:
            out[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
        return out

    for mid, sub in te.groupby('molecule_id', sort=False):
        if mid not in BASE or len(BASE[mid][0]) < 2:
            continue
        smis, keys, lib_max, scs, forms = BASE[mid]
        spectra = [dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, prec=r.precursor_mz, adduct=r.adduct,
                        mode=1 if r.ionization_mode == 'positive' else -1,
                        tims=_has_inst and str(r.instrument_type).lower().replace('-', '').replace(' ', '') == 'timstof')
                   for r in sub.itertuples()]
        s = _cm.score_candidates(_cm_model, spectra, list(smis), dev)
        comb = _z(scs) + CM_LAM * _z(s)
        order = list(range(len(smis)))
        groups = {}
        for i, f in enumerate(forms):
            groups.setdefault(str(f).split('|')[0], []).append(i)
        for idx in groups.values():
            if len(idx) > 1:
                for slot, j in zip(idx, sorted(idx, key=lambda j: -comb[j])):
                    order[slot] = j
        cm_stats['molecules'] += 1
        cm_stats['changed_top1'] += int(order[0] != 0)
        cm_stats['changed_top25'] += int(order[:25] != list(range(min(25, len(order)))))
        BASE[mid] = ([smis[i] for i in order], [keys[i] for i in order], lib_max, [float(comb[i]) for i in order],
                     [forms[i] for i in order])
    print('CMatch re-rank', cm_stats, f'{time.time() - T0:.0f}s')
except Exception as e:
    print('CMATCH SKIPPED -> 0.401 behaviour:', repr(e))

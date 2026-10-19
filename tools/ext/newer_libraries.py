# Count (and keep) public library spectra of compounds NOT in the competition train set, from the newest GNPS, MassBank
# and MoNA releases (host: newer releases of these libraries may be used, CC BY-SA included; NC "should be fine").
# Internet ON, CPU. Each source is optional: a failed download is reported and skipped. Writes
#   /kaggle/working/newer_lib_report.json   counts per source (spectra, compounds, not-in-train, instrument, mode, adduct, licence)
#   /kaggle/working/newer_lib_unseen.parquet  spectra of not-in-train compounds (test.parquet-like layout + truth + provenance)
# The parquet is derived from train membership -> save it only as a PRIVATE dataset. Never commit outputs.
import glob, json, os, re, subprocess, sys, time, zipfile
import numpy as np, pandas as pd, pyarrow.parquet as pq, requests

T0 = time.time()
W, TMP = '/kaggle/working', '/tmp/newlib'
os.makedirs(TMP, exist_ok=True)
try:
    from rdkit import Chem
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'], check=False)
    from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import inchi as rdinchi
RDLogger.DisableLog('rdApp.*')


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def fetch(urls, dest):
    for u in urls:
        try:
            with requests.get(u, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for ch in r.iter_content(1 << 22):
                        f.write(ch)
            log(f'downloaded {u}: {os.path.getsize(dest) / 1e6:.0f} MB')
            if zipfile.is_zipfile(dest):
                with zipfile.ZipFile(dest) as z:
                    name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
                    z.extract(name, TMP)
                os.remove(dest)
                os.rename(os.path.join(TMP, name), dest)
                log(f'unzipped {name}: {os.path.getsize(dest) / 1e6:.0f} MB')
            return u
        except Exception as e:
            log('download failed', u, repr(e)[:200])
    return None


# ---------------- competition train set ----------------
TR = sorted(glob.glob('/kaggle/input/**/train.parquet', recursive=True), key=len)[0]
SCH = pq.read_schema(TR)
log('train schema:', [(f.name, str(f.type)) for f in SCH])
TRAIN_IK = {k for k in pq.read_table(TR, columns=['inchikey14']).column(0).to_pylist() if k}
log(f'train compounds (InChIKey-14): {len(TRAIN_IK):,}')
ID_RE = re.compile(r'^(CCMSLIB\d+|MSBNK-[\w-]+|MoNA\d+|[A-Z]{2,8}\d{5,})$')
TRAIN_IDS, id_cols = set(), []
for f in SCH:
    if f.name != 'inchikey14' and str(f.type) in ('string', 'large_string'):
        v = pq.ParquetFile(TR).read_row_group(0, columns=[f.name]).column(0).to_pylist()[:5000]
        if sum(isinstance(x, str) and bool(ID_RE.match(x)) for x in v) > 50:
            id_cols.append(f.name)
            TRAIN_IDS |= {x for x in pq.read_table(TR, columns=[f.name]).column(0).to_pylist() if x}
log('train columns holding library record ids:', id_cols, f'({len(TRAIN_IDS):,} ids)')

# ---------------- normalisers ----------------
STD = {'[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-'}
TEST_ADDUCTS = STD | {'[M-H2O+H]+', '[M-2H2O+H]+', '[M-H2O-H]-'}
ALIAS = {'[M+FA-H]-': '[M+CH2O2-H]-', '[M+HCOO]-': '[M+CH2O2-H]-', '[M+HCOOH-H]-': '[M+CH2O2-H]-',
         '[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+', '[M-H-H2O]-': '[M-H2O-H]-'}


def norm_mode(m, adduct=None):
    m = (m or '').strip().lower()
    if m.startswith('p') or m == '+':
        return 'positive'
    if m.startswith('n') or m == '-':
        return 'negative'
    a = (adduct or '').strip()
    return 'positive' if a.endswith('+') else 'negative' if a.endswith('-') else None


def norm_adduct(a, mode):
    if not isinstance(a, str) or not a.strip():
        return None
    a = a.strip().replace(' ', '')
    a = re.sub(r'\](\d*)([+-])$', lambda m: ']' + ('' if m.group(1) in ('', '1') else m.group(1)) + m.group(2), a)
    if not a.startswith('['):
        core, sign = a, ''
        if re.match(r'^M.*[A-Za-z0-9)][+-]$', core):
            core, sign = core[:-1], core[-1]
        sign = sign or ('+' if mode == 'positive' else '-' if mode == 'negative' else '')
        a = f'[{core}]{sign}'
    return ALIAS.get(a, a)


def inst_class(s):
    s = (s or '').lower()
    if 'tims' in s:
        return 'timsTOF'
    if any(k in s for k in ('orbitrap', 'exactive', 'exploris', 'fusion', 'lumos', 'astral', 'ft-icr', 'fticr', 'velos',
                            'elite', 'itft', 'qft', 'ltq-ft')):
        return 'Orbitrap/FT'
    if any(k in s for k in ('tof', 'maxis', 'impact', 'compact', 'synapt', 'xevo', '5600', '6600', '7600', '6530', '6540',
                            '6545', '6550', '6560')):
        return 'QTOF'
    return 'unknown' if s.strip() in ('', 'nan', 'none', 'n/a', 'na') else 'other'


_ikc = {}


def ik14_of(ik, smi, inc):
    ik = (ik or '').strip().strip('"')
    if re.match(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$', ik):
        return ik[:14]
    smi, inc = (smi or '').strip().strip('"'), (inc or '').strip().strip('"')
    key = (smi, inc)
    if key not in _ikc:
        out = None
        try:
            m = rdinchi.MolFromInchi(inc) if inc.startswith('InChI=') else None
            if m is None and smi and smi.upper() not in ('N/A', 'NA', 'NONE'):
                m = Chem.MolFromSmiles(smi)
            out = (Chem.MolToInchiKey(m)[:14] or None) if m is not None else None
        except Exception:
            out = None
        _ikc[key] = out
    return _ikc[key]


def num(s):
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s or ''))
    return float(m.group()) if m else np.nan


ROWS, PEAKS = [], {}


def add(source, rec_id, ik, smi, inc, prec, adduct, mode, inst, ce, lic, lib, mz, it):
    mode = norm_mode(mode, adduct)
    k = ik14_of(ik, smi, inc)
    unseen = k is not None and k not in TRAIN_IK
    ROWS.append(dict(source=source, rec_id=rec_id, ik14=k, unseen=unseen, precursor_mz=num(prec),
                     adduct=norm_adduct(adduct, mode), mode=mode, instrument=inst, inst_class=inst_class(inst),
                     ce=ce, license=lic, library=lib, n_peaks=len(mz), id_in_train=bool(rec_id) and rec_id in TRAIN_IDS))
    if unseen and len(mz):
        PEAKS[len(ROWS) - 1] = (smi, inc, mz, it)


# ---------------- parsers ----------------
def parse_mgf(path):
    rec, mz, it = None, [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln == 'BEGIN IONS':
                rec, mz, it = {}, [], []
            elif ln == 'END IONS':
                if rec is not None:
                    yield rec, mz, it
                rec = None
            elif rec is not None:
                if '=' in ln and not ln[0].isdigit():
                    k, v = ln.split('=', 1)
                    rec[k.strip().upper()] = v.strip()
                else:
                    p = ln.split()
                    try:
                        mz.append(float(p[0])); it.append(float(p[1]))
                    except (ValueError, IndexError):
                        pass


def parse_msp(path):
    rec, mz, it, inpk = {}, [], [], False
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                if rec:
                    yield rec, mz, it
                rec, mz, it, inpk = {}, [], [], False
                continue
            if inpk:
                for a, b in re.findall(r'([\d.]+)[\s:,]+([\d.eE+-]+)', s):
                    try:
                        mz.append(float(a)); it.append(float(b))
                    except ValueError:
                        pass
                continue
            if ':' in s:
                k, v = s.split(':', 1)
                k, v = k.strip().upper(), v.strip()
                if k == 'NUM PEAKS':
                    inpk = True
                else:
                    rec[k] = rec[k] + ' | ' + v if k in rec else v
    if rec:
        yield rec, mz, it


def g(rec, *keys):
    for k in keys:
        v = rec.get(k)
        if v not in (None, '', 'N/A', 'NA', 'null'):
            return v
    return None


def comment_field(rec, name):
    m = re.search(r'"?' + name + r'=([^"|]+)"?', rec.get('COMMENTS', '') + ' ' + rec.get('COMMENT', ''), re.I)
    return m.group(1).strip() if m else None


def ms2_only(rec):
    t = (g(rec, 'MSLEVEL', 'SPECTRUM_TYPE', 'SPECTRUMTYPE', 'MS_LEVEL') or '2').upper()
    return t in ('2', 'MS2')


SRC = {}
# ---------------- GNPS (community libraries, no propagated spectra) ----------------
p = os.path.join(TMP, 'gnps.mgf')
u = fetch(['https://external.gnps2.org/gnpslibrary/ALL_GNPS_NO_PROPOGATED.mgf',
           'https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS_NO_PROPOGATED.mgf'], p)
SRC['GNPS'] = u
if u:
    n0 = len(ROWS)
    for rec, mz, it in parse_mgf(p):
        if not ms2_only(rec):
            continue
        name = rec.get('NAME', '')
        add_ = rec.get('ADDUCT') or (name.rsplit(' ', 1)[-1] if ' ' in name else None)
        if add_ and not re.match(r'^\[?M', add_):
            add_ = None
        add('GNPS', rec.get('SPECTRUMID'), rec.get('INCHIAUX') if str(rec.get('INCHIAUX', '')).count('-') == 2 else None,
            rec.get('SMILES'), rec.get('INCHI'), rec.get('PEPMASS'), add_, rec.get('IONMODE'), rec.get('SOURCE_INSTRUMENT'),
            rec.get('COLLISION_ENERGY'), 'GNPS library (per sub-library; community libraries CC0)', rec.get('ORGANISM'), mz, it)
    log(f'GNPS: {len(ROWS) - n0:,} MS2 spectra parsed')
    os.remove(p)

# ---------------- MassBank (latest GitHub release) ----------------
try:
    rel = requests.get('https://api.github.com/repos/MassBank/MassBank-data/releases/latest', timeout=60).json()
    assets = {a['name']: a['browser_download_url'] for a in rel.get('assets', [])}
    log('MassBank release', rel.get('tag_name'), list(assets))
except Exception as e:
    rel, assets = {}, {}
    log('MassBank release lookup failed', repr(e)[:200])
MB_LIC = {}
p = os.path.join(TMP, 'massbank.json')
if 'MassBank.json' in assets and fetch([assets['MassBank.json']], p):     # licences per accession
    try:
        data = json.load(open(p, encoding='utf-8'))
        data = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])

        def _walk(o, path=''):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from _walk(v, path + '/' + str(k))
            elif isinstance(o, list):
                for v in o:
                    yield from _walk(v, path)
            elif isinstance(o, str):
                yield path, o
        for i, r in enumerate(data):
            acc = lic = None
            for path, v in _walk(r):
                if acc is None and v.startswith('MSBNK-'):
                    acc = v
                if lic is None and 'licen' in path.lower():
                    lic = v
            if acc:
                MB_LIC[acc] = lic
            if i == 0:
                log('MassBank.json first record keys:', list(r)[:25] if isinstance(r, dict) else type(r))
        log(f'MassBank licences read for {len(MB_LIC):,} accessions')
        del data
    except Exception as e:
        log('MassBank.json parse failed', repr(e)[:200])
    os.remove(p)
p = os.path.join(TMP, 'massbank.msp')
mb_url = fetch([assets[n] for n in ('MassBank_RIKENformat.msp', 'MassBank_NISTformat.msp') if n in assets], p)
SRC['MassBank'] = (rel.get('tag_name'), mb_url)
if mb_url:
    n0, first = len(ROWS), True
    for rec, mz, it in parse_msp(p):
        if first:
            log('MassBank msp first record fields:', {k: v[:60] for k, v in rec.items()}); first = False
        if not ms2_only(rec):
            continue
        acc = next((m.group() for v in rec.values() for m in [re.search(r'MSBNK-[\w-]+', v)] if m), None)
        add('MassBank', acc, g(rec, 'INCHIKEY'), g(rec, 'SMILES') or comment_field(rec, 'SMILES'), g(rec, 'INCHI'),
            g(rec, 'PRECURSORMZ'), g(rec, 'PRECURSORTYPE', 'PRECURSOR_TYPE'), g(rec, 'IONMODE', 'ION_MODE'),
            ' '.join(x for x in (g(rec, 'INSTRUMENTTYPE', 'INSTRUMENT_TYPE'), g(rec, 'INSTRUMENT')) if x),
            g(rec, 'COLLISIONENERGY', 'COLLISION_ENERGY'), MB_LIC.get(acc) or comment_field(rec, 'license'),
            acc.split('-')[1] if acc and acc.count('-') >= 2 else None, mz, it)
    log(f'MassBank: {len(ROWS) - n0:,} MS2 spectra parsed')
    os.remove(p)

# ---------------- MoNA (predefined exports API; skipped if unavailable) ----------------
mona_url = None
try:
    ex = requests.get('https://mona.fiehnlab.ucdavis.edu/rest/downloads/predefined', timeout=60).json()
    log('MoNA exports:', [(e.get('label'), e.get('query', '')[:60]) for e in ex])
    pick = [e for e in ex if re.search(r'LC-MS/MS|LC-MS-MS|Experimental', str(e.get('label')), re.I)]
    pick = pick or ex
    for e in pick:
        for fmt in ('mspExport', 'jsonExport'):
            exp = e.get(fmt) or {}
            if exp.get('id') and fmt == 'mspExport':
                mona_url = f"https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/{exp['id']}"
                break
        if mona_url:
            log('MoNA export chosen:', e.get('label'), (e.get('mspExport') or {}).get('size')); break
except Exception as e:
    log('MoNA export list failed', repr(e)[:200])
p = os.path.join(TMP, 'mona.msp')
SRC['MoNA'] = fetch([mona_url], p) if mona_url else None
if SRC['MoNA']:
    n0, first = len(ROWS), True
    for rec, mz, it in parse_msp(p):
        if first:
            log('MoNA msp first record fields:', {k: v[:60] for k, v in rec.items()}); first = False
        if not ms2_only(rec):
            continue
        add('MoNA', g(rec, 'DB#', 'ID'), g(rec, 'INCHIKEY'), comment_field(rec, 'SMILES') or g(rec, 'SMILES'),
            comment_field(rec, 'InChI') or g(rec, 'INCHI'), g(rec, 'PRECURSORMZ'), g(rec, 'PRECURSOR_TYPE', 'PRECURSORTYPE'),
            g(rec, 'ION_MODE', 'IONMODE'), ' '.join(x for x in (g(rec, 'INSTRUMENT_TYPE'), g(rec, 'INSTRUMENT')) if x),
            g(rec, 'COLLISION_ENERGY'), comment_field(rec, 'license'), comment_field(rec, 'submitter'), mz, it)
    log(f'MoNA: {len(ROWS) - n0:,} MS2 spectra parsed')
    os.remove(p)

# ---------------- report ----------------
df = pd.DataFrame(ROWS)
log(f'all sources: {len(df):,} spectra; structure keys computed for {len(_ikc):,} records without an InChIKey')
top = lambda s, n=12: {str(k): int(v) for k, v in s.value_counts().head(n).items()}
rep = dict(sources=SRC, train_compounds=len(TRAIN_IK), train_id_columns=id_cols)
for src, d in list(df.groupby('source')) + [('ALL', df)]:
    u = d[d.unseen]
    good = u[u.adduct.isin(STD) & u.inst_class.isin(['timsTOF', 'QTOF'])]
    rep[src] = dict(spectra=len(d), with_structure=int(d.ik14.notna().sum()), compounds=int(d.ik14.nunique()),
                    record_ids_already_in_train=int(d.id_in_train.sum()) if TRAIN_IDS else None,
                    unseen_compounds=int(u.ik14.nunique()), unseen_spectra=len(u),
                    unseen_compounds_by_instrument={k: int(v) for k, v in u.groupby('inst_class').ik14.nunique().items()},
                    unseen_compounds_by_mode={str(k): int(v) for k, v in u.groupby('mode').ik14.nunique().items()},
                    unseen_compounds_with_test_adduct=int(u[u.adduct.isin(TEST_ADDUCTS)].ik14.nunique()),
                    unseen_compounds_std_adduct_qtof_timstof=int(good.ik14.nunique()),
                    unseen_compounds_std_adduct_timstof=int(good[good.inst_class == 'timsTOF'].ik14.nunique()),
                    unseen_top_adducts=top(u.adduct), unseen_top_instruments=top(u.instrument),
                    unseen_top_libraries=top(u.library), unseen_licences=top(u.license.fillna('unknown')))
json.dump(rep, open(os.path.join(W, 'newer_lib_report.json'), 'w'), indent=1, default=str)
print('NEWER LIB REPORT ' + json.dumps(rep, default=str))

# ---------------- unseen-compound spectra (private) ----------------
if PEAKS:
    idx = np.array(sorted(PEAKS))
    out = df.iloc[idx].reset_index(drop=True)
    out['smiles'] = [PEAKS[i][0] for i in idx]
    out['inchi'] = [PEAKS[i][1] for i in idx]
    out['ms2_mzs'] = [list(map(float, PEAKS[i][2])) for i in idx]
    its = [np.asarray(PEAKS[i][3], float) for i in idx]
    out['ms2_normalized_intensities'] = [list(x / x.max()) if len(x) and x.max() > 0 else list(x) for x in its]
    out = out.rename(columns={'ik14': 'inchikey14', 'mode': 'ionization_mode', 'instrument': 'instrument_type',
                              'ce': 'collision_energy_orig'})
    out.to_parquet(os.path.join(W, 'newer_lib_unseen.parquet'), index=False)
    log(f'kept {len(out):,} spectra of {out.inchikey14.nunique():,} not-in-train compounds -> newer_lib_unseen.parquet')
log('done')

"""CMatch: contrastive spectrum <-> molecule matcher (JESTR/MSAlign-style, trained from scratch on competition data).

Shared by the training notebook and the submission notebook, so featurization is identical in both.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PEAKS = 64
MORGAN_BITS = 2048
MOL_DIM = MORGAN_BITS + 167
ADDUCTS = ['[M+H]+', '[M+Na]+', '[M+NH4]+', '[M+K]+', '[M-H2O+H]+', '[M+H-H2O]+', '[M]+', '[M-H]-', '[M+Cl]-',
           '[M+FA-H]-', '[M+HCOO]-', '[M-H2O-H]-', '[M+CH3COO]-', '[M]-']
ADDUCT_ID = {a: i + 1 for i, a in enumerate(ADDUCTS)}   # 0 = other / unknown


def adduct_id(a):
    return ADDUCT_ID.get(str(a).replace(' ', ''), 0)


def spectrum_arrays(mz, it, k=K_PEAKS):
    """Top-k peaks by intensity -> (k,) m/z and (k,) intensity (max-normalised), zero padded."""
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    out_mz = np.zeros(k, np.float32)
    out_it = np.zeros(k, np.float32)
    if len(mz):
        o = np.argsort(-it, kind='stable')[:k]
        out_mz[:len(o)] = mz[o]
        out_it[:len(o)] = it[o] / it[o].max()
    return out_mz, out_it


def mol_features(smiles):
    """Morgan r=2 counts (2048, clipped to 255) + MACCS keys (167) as uint8; None if RDKit can't parse it."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    if 'gen' not in mol_features.__dict__:
        mol_features.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    cnt = np.minimum(mol_features.gen.GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
    mac = np.zeros(167, np.uint8)
    for b in MACCSkeys.GenMACCSKeys(m).GetOnBits():
        mac[b] = 1
    return np.concatenate([cnt, mac])


class SinEmb(nn.Module):
    def __init__(self, n=32, lo=0.01, hi=2000.0):
        super().__init__()
        self.register_buffer('w', 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n))

    def forward(self, x):
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)


class SpecEncoder(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, out=256, drop=0.1):
        super().__init__()
        self.sin = SinEmb()
        self.peak = nn.Linear(64 * 2 + 1, d)
        self.prec = nn.Linear(64, d)
        self.add = nn.Embedding(len(ADDUCTS) + 1, d)
        self.mode = nn.Embedding(2, d)
        self.tims = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out)

    def forward(self, mz, it, prec, add, mode, tims):
        pad = it <= 0
        nl = (prec.unsqueeze(1) - mz).clamp(min=0)
        tok = self.peak(torch.cat([self.sin(mz), self.sin(nl), it.sqrt().unsqueeze(-1)], -1))
        cls = self.prec(self.sin(prec)) + self.add(add) + self.mode(mode) + self.tims(tims)
        x = torch.cat([cls.unsqueeze(1), tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        x = self.enc(x, src_key_padding_mask=mask)
        return F.normalize(self.head(self.norm(x[:, 0])), dim=-1)


class MolEncoder(nn.Module):
    def __init__(self, out=256, hid=1024, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MOL_DIM, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, out))

    def forward(self, x):
        return F.normalize(self.net(torch.log1p(x.float())), dim=-1)


class CMatch(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, out=256):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, out=out)
        self.spec = SpecEncoder(d, layers, heads, out)
        self.mol = MolEncoder(out)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def save(self, path, **extra):
        torch.save(dict(cfg=self.cfg, state=self.state_dict(), **extra), path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(**ck['cfg'])
        m.load_state_dict(ck['state'])
        return m.to(device).eval(), ck


@torch.no_grad()
def embed_spectra(model, spectra, device, bs=256):
    """spectra: list of dicts with mz, it, prec, adduct, mode (+1/-1), tims (bool) -> (n, out) tensor."""
    outs = []
    for i in range(0, len(spectra), bs):
        chunk = spectra[i:i + bs]
        arr = [spectrum_arrays(s['mz'], s['it']) for s in chunk]
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        outs.append(model.spec(t([a[0] for a in arr]), t([a[1] for a in arr]), t([float(s['prec']) for s in chunk]),
                               t([adduct_id(s['adduct']) for s in chunk], torch.long),
                               t([int(s['mode'] == 1) for s in chunk], torch.long),
                               t([int(bool(s.get('tims', False))) for s in chunk], torch.long)))
    return torch.cat(outs) if outs else torch.zeros(0, model.cfg['out'], device=device)


@torch.no_grad()
def score_candidates(model, spectra, smiles, device):
    """Cosine similarity of each candidate SMILES to the mean embedding of a molecule's spectra (nan if unparsable)."""
    if not spectra or not smiles:
        return np.full(len(smiles), np.nan, np.float32)
    q = F.normalize(embed_spectra(model, spectra, device).mean(0, keepdim=True), dim=-1)
    feats = [mol_features(s) for s in smiles]
    ok = [i for i, f in enumerate(feats) if f is not None]
    out = np.full(len(smiles), np.nan, np.float32)
    if ok:
        m = model.mol(torch.as_tensor(np.stack([feats[i] for i in ok]), device=device))
        out[ok] = (m @ q.T).squeeze(1).float().cpu().numpy()
    return out

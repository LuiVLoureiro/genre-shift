# -*- coding: utf-8 -*-
"""Validação da v14 — até onde estender a FRAÇÃO-ALVO da seleção?

No split por pseudo-música: seleção completa estilo v13 (consenso híbrido +
individual, cotas por classe) em frações-alvo crescentes; mede a acurácia
REAL do pool selecionado em cada fração. Aprovar a maior fração cuja
acurácia do pool não caia mais que ~0,5pp vs a fração 0,85 atual."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_comum import carregar, criar_features, ajustar_tratamento, aplicar_tratamento
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RS = 42
LIMIAR, CONF_MINIMA = 0.60, 0.70
df_tr, _ = carregar()
y = df_tr["GENRE"].values
n_tr = len(y)
classes_ref = np.unique(y)
y_int = np.searchsorted(classes_ref, y)

X_fe = criar_features(df_tr.drop(columns=["GENRE"]), com_dct=True, com_geo=False)
Z = StandardScaler().fit_transform(df_tr.drop(columns=["GENRE"]).values.astype(np.float32))
E = PCA(n_components=0.95, random_state=RS).fit_transform(Z).astype(np.float32)

# grupos híbridos (kNN-mútuo k=3 -> fino), como na v13
km_fino = KMeans(n_clusters=int(round(n_tr / 2.5)), n_init=1, max_iter=100,
                 random_state=RS).fit(E)
rot_fino = km_fino.labels_
nn = NearestNeighbors(n_neighbors=4).fit(E)
dist, viz = nn.kneighbors(E)
teto = np.median(dist[:, 1]) * 1.5
conj = [set(viz[i, 1:]) for i in range(n_tr)]
li_, co_ = [], []
for i in range(n_tr):
    for p in range(1, 4):
        j = viz[i, p]
        if dist[i, p] <= teto and i in conj[j]:
            li_.append(i); co_.append(j)
G = coo_matrix((np.ones(len(li_)), (li_, co_)), shape=(n_tr, n_tr))
_, comp = connected_components(G, directed=False)
tam = np.bincount(comp)
rot = np.where(tam[comp] >= 2, comp, comp.max() + 1 + rot_fino)
_, rot = np.unique(rot, return_inverse=True)

pm = KMeans(n_clusters=700, n_init=1, max_iter=100, random_state=RS).fit(E).labels_
gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=RS)

for fold, (ia, ib) in enumerate(gss.split(X_fe, y, groups=pm)):
    cols, li, ls = ajustar_tratamento(X_fe.iloc[ia])
    A = aplicar_tratamento(X_fe.iloc[ia], cols, li, ls)
    B = aplicar_tratamento(X_fe.iloc[ib], cols, li, ls)
    sc = StandardScaler()
    A, B = sc.fit_transform(A), sc.transform(B)
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=RS).fit(A, y[ia])
    prob_b = hgb.predict_proba(B)
    pred_i, conf_i = prob_b.argmax(1), prob_b.max(1)
    n_b = len(ib)

    # consenso híbrido (colegas de holdout)
    cl_b = rot[ib]
    k = rot.max() + 1
    S = np.zeros((k, 6)); C = np.zeros(k)
    np.add.at(S, cl_b, prob_b); np.add.at(C, cl_b, 1.0)
    media = S[cl_b] / np.maximum(C[cl_b], 1)[:, None]
    c_lab, c_conf = media.argmax(1), media.max(1)
    usa = (C[cl_b] >= 2) & (c_conf >= LIMIAR)
    lab_cand = np.where(usa, c_lab, pred_i)
    conf_cand = np.where(usa, c_conf, conf_i)
    piso = np.where(usa, LIMIAR, CONF_MINIMA)
    prior_est = prob_b.mean(axis=0)          # proxy dos priors estimados

    print(f"\n===== FOLD {fold + 1} =====")
    print(f"{'fração':>7} | {'selecionados':>12} | {'acc do pool':>11} | {'acc do incremento':>17}")
    sel_ant = None
    for frac in [0.85, 0.88, 0.90, 0.92, 0.95]:
        n_alvo = frac * n_b
        sel = np.concatenate([
            (ic := np.where((lab_cand == i) & (conf_cand >= piso))[0])
            [np.argsort(-conf_cand[ic])[:int(round(prior_est[i] * n_alvo))]]
            for i in range(6)])
        acc = (lab_cand[sel] == y_int[ib][sel]).mean()
        if sel_ant is not None:
            novos = np.setdiff1d(sel, sel_ant)
            acc_inc = (lab_cand[novos] == y_int[ib][novos]).mean() if len(novos) else float("nan")
            print(f"{frac:>7.0%} | {len(sel):>12} | {acc:>11.4f} | {acc_inc:>13.2%} ({len(novos)} novos)")
        else:
            print(f"{frac:>7.0%} | {len(sel):>12} | {acc:>11.4f} | {'—':>17}")
        sel_ant = sel

print("\nValidação de frações concluída.")

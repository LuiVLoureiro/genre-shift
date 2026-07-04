# -*- coding: utf-8 -*-
"""Pipeline comum: FE (com DCT opcional) + geometria espectral contínua (GEO_*)
+ tratamento de outliers."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson
try:
    from scipy.integrate import cumulative_simpson          # scipy >= 1.12
except ImportError:                                          # fallback robusto
    from scipy.integrate import cumulative_trapezoid
    def cumulative_simpson(y, x=None, axis=-1, initial=0.0):
        return cumulative_trapezoid(y, x=x, axis=axis, initial=initial)

EPS = 1e-9

# chave global do experimento de geometria espectral (GEO_*).
# False preserva o comportamento arquivado de gerar_v5/gerar_v6.
USAR_GEOMETRIA_ESPECTRAL = False
N_GRADE_GEO = 201   # grade fina (ímpar) em [0,1] para interpolação/integração

COLS_ASE  = [f"PAR_ASE{i}"  for i in range(1, 35)]
COLS_ASEV = [f"PAR_ASEV{i}" for i in range(1, 35)]
COLS_SFM  = [f"PAR_SFM{i}"  for i in range(1, 25)]
COLS_SFMV = [f"PAR_SFMV{i}" for i in range(1, 25)]
COLS_MFCC  = [f"PAR_MFCC{i}"  for i in range(1, 21)]
COLS_MFCCV = [f"PAR_MFCCV{i}" for i in range(1, 21)]


def inclinacao_bandas(m):
    x = np.arange(m.shape[1])
    return np.polyfit(x, m.T, 1)[0]


def coeficientes_dct(matriz, n_coefs, log=False):
    """DCT-II ao longo das bandas: descreve a FORMA do espectro em coeficientes
    ordenados do padrao grosso (baixa ordem) ao detalhe fino (alta ordem)."""
    m = np.log1p(np.abs(matriz)) * np.sign(matriz) if log else matriz
    return dct(m, type=2, axis=1, norm="ortho")[:, :n_coefs]


# ======================================================================
# GEOMETRIA ESPECTRAL CONTÍNUA (features GEO_*)
# Cada vetor de bandas é tratado como uma DISTRIBUIÇÃO DE ENERGIA p(x)
# em x ∈ [0,1] (grave -> agudo). Todas as operações são POR LINHA:
# nenhuma estatística cruza amostras => zero vazamento por construção.
# ======================================================================

def _densidade_espectral(matriz, n_grade=N_GRADE_GEO):
    """Converte cada linha em densidade contínua p(x) >= 0 com ∫p dx = 1.

    1) desloca ao mínimo da linha (invariância a nível/volume/masterização);
    2) interpola com PCHIP — cúbica monótona por trechos, sem as oscilações
       de Runge de um polinômio global;
    3) normaliza pela integral de Simpson (erro O(h^4)).
    Devolve (xg, p, dp): grade, densidades e derivadas p'(x) por linha."""
    n, b = matriz.shape
    x0 = np.linspace(0.0, 1.0, b)
    xg = np.linspace(0.0, 1.0, n_grade)
    e = matriz.astype(np.float64) - np.nanmin(matriz, axis=1, keepdims=True)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    pch = PchipInterpolator(x0, e, axis=1)
    pg = np.clip(np.nan_to_num(pch(xg)), 0.0, None)          # (n, grade)
    dg = np.nan_to_num(pch.derivative()(xg))
    Z = simpson(pg, x=xg, axis=1)
    quase_nulo = Z < 1e-12                                    # espectro ~constante
    if quase_nulo.any():                                      # -> densidade uniforme
        pg[quase_nulo], dg[quase_nulo] = 1.0, 0.0
        Z = simpson(pg, x=xg, axis=1)
    return xg, pg / Z[:, None], dg / Z[:, None]


def _interp_por_linha(xg, M, q):
    """Interpolação linear de M[i, :] (função na grade xg) no ponto q[i]."""
    pos = np.clip(np.searchsorted(xg, q) - 1, 0, len(xg) - 2)
    t = np.clip((q - xg[pos]) / (xg[pos + 1] - xg[pos]), 0.0, 1.0)
    fila = np.arange(M.shape[0])
    return M[fila, pos] + t * (M[fila, pos + 1] - M[fila, pos])


def _quantil_espectral(xg, p, F, alpha, max_newton=12, tol=1e-4):
    """Resolve ∫_0^q p(x) dx = alpha  (i.e., F(q) = alpha) por linha.

    Newton-Raphson vetorizado: q <- q - (F(q) - alpha)/p(q), pois F' = p.
    Fallback por BISSEÇÃO (F é monótona não decrescente, logo a bisseção
    sempre converge) quando Newton sai de [0,1], encontra p(q) ~ 0 ou não
    atinge a tolerância."""
    n = p.shape[0]
    q = np.full(n, 0.5)
    estavel = np.ones(n, bool)
    for _ in range(max_newton):
        Fq = _interp_por_linha(xg, F, q)
        pq = _interp_por_linha(xg, p, q)
        q_novo = q - (Fq - alpha) / np.maximum(pq, 1e-9)
        ruim = (~np.isfinite(q_novo)) | (q_novo < 0.0) | (q_novo > 1.0) | (pq < 1e-6)
        estavel &= ~ruim
        q = np.where(estavel, np.clip(q_novo, 0.0, 1.0), q)
    falhou = (~estavel) | (np.abs(_interp_por_linha(xg, F, q) - alpha) > tol)
    if falhou.any():
        Ff = F[falhou]
        lo, hi = np.zeros(falhou.sum()), np.ones(falhou.sum())
        for _ in range(40):                                   # 2^-40: precisão sobrada
            mid = 0.5 * (lo + hi)
            abaixo = _interp_por_linha(xg, Ff, mid) < alpha
            lo, hi = np.where(abaixo, mid, lo), np.where(abaixo, hi, mid)
        q[falhou] = 0.5 * (lo + hi)
    return q


def _geometria_espectral(matriz, prefixo):
    """Momentos e quantis da densidade espectral contínua de cada linha."""
    xg, p, dp = _densidade_espectral(matriz)
    dx = xg[None, :]
    mu = simpson(p * dx, x=xg, axis=1)                        # centroide ∫x p dx
    var = np.clip(simpson(p * (dx - mu[:, None]) ** 2, x=xg, axis=1), 0.0, None)
    desvio = np.sqrt(var)
    m3 = simpson(p * (dx - mu[:, None]) ** 3, x=xg, axis=1)   # 3º momento central
    assimetria = m3 / np.maximum(desvio ** 3, 1e-9)
    entropia = -simpson(p * np.log(p + 1e-12), x=xg, axis=1)  # entropia diferencial
    pico = xg[p.argmax(axis=1)]                               # moda da densidade
    var_total = simpson(np.abs(dp), x=xg, axis=1)             # ∫|p'| dx: rugosidade
    F = cumulative_simpson(p, x=xg, axis=1, initial=0.0)      # F(x) = ∫_0^x p
    F = F / np.maximum(F[:, -1:], 1e-12)                      # garante F(1) = 1
    q25 = _quantil_espectral(xg, p, F, 0.25)
    q50 = _quantil_espectral(xg, p, F, 0.50)
    q75 = _quantil_espectral(xg, p, F, 0.75)
    feats = {
        f"GEO_{prefixo}_CENTROIDE": mu,
        f"GEO_{prefixo}_DESVIO": desvio,
        f"GEO_{prefixo}_ASSIMETRIA": assimetria,
        f"GEO_{prefixo}_ENTROPIA": entropia,
        f"GEO_{prefixo}_PICO": pico,
        f"GEO_{prefixo}_VARIACAO_TOTAL": var_total,
        f"GEO_{prefixo}_MEDIANA": q50,
        f"GEO_{prefixo}_IQR": q75 - q25,
    }
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in feats.items()}


def criar_features(df, com_dct=True, n_dct=10, com_geo=None):
    novo = df.copy()
    ase, asev = df[COLS_ASE].values, df[COLS_ASEV].values
    sfm, sfmv = df[COLS_SFM].values, df[COLS_SFMV].values
    mfcc, mfccv = df[COLS_MFCC].values, df[COLS_MFCCV].values

    novo["FE_ASE_STD"] = ase.std(axis=1)
    novo["FE_ASE_AMPLITUDE"] = ase.max(axis=1) - ase.min(axis=1)
    novo["FE_ASE_INCLINACAO"] = inclinacao_bandas(ase)
    grave, media, aguda = ase[:, :11].mean(1), ase[:, 11:23].mean(1), ase[:, 23:].mean(1)
    novo["FE_ASE_GRAVE"], novo["FE_ASE_MEDIA"], novo["FE_ASE_AGUDA"] = grave, media, aguda
    novo["FE_ASE_GRAVE_AGUDO"] = grave / (aguda + np.sign(aguda) * EPS + EPS)
    novo["FE_ASEV_STD"] = asev.std(axis=1)
    novo["FE_ASEV_INCLINACAO"] = inclinacao_bandas(asev)
    novo["FE_SFM_STD"] = sfm.std(axis=1)
    novo["FE_SFM_INCLINACAO"] = inclinacao_bandas(sfm)
    sg, sm, sa = sfm[:, :8].mean(1), sfm[:, 8:16].mean(1), sfm[:, 16:].mean(1)
    novo["FE_SFM_GRAVE"], novo["FE_SFM_MEDIA"], novo["FE_SFM_AGUDA"] = sg, sm, sa
    novo["FE_SFM_GRAVE_AGUDO"] = sg / (sa + EPS)
    novo["FE_MFCC_STD"] = mfcc.std(axis=1)
    novo["FE_MFCCV_MEDIA"] = mfccv.mean(axis=1)
    novo["FE_MFCC_INSTABILIDADE"] = mfccv.mean(1) / (np.abs(mfcc).mean(1) + EPS)
    novo["FE_SC_CV"] = df["PAR_SC_V"] / (df["PAR_SC"].abs() + EPS)
    novo["FE_ASC_CV"] = df["PAR_ASC_V"] / (df["PAR_ASC"].abs() + EPS)
    novo["FE_ASS_CV"] = df["PAR_ASS_V"] / (df["PAR_ASS"].abs() + EPS)
    novo["FE_THR21"] = df["PAR_THR_2RMS_TOT"] / (df["PAR_THR_1RMS_TOT"] + EPS)
    novo["FE_THR31"] = df["PAR_THR_3RMS_TOT"] / (df["PAR_THR_1RMS_TOT"] + EPS)
    novo["FE_TCD31"] = df["PAR_3RMS_TCD"] / (df["PAR_1RMS_TCD"] + EPS)

    if com_dct:
        for nome, matriz, log in [("ASE", ase, False), ("ASEV", asev, True),
                                  ("SFM", sfm, False), ("SFMV", sfmv, True)]:
            n = min(n_dct, matriz.shape[1])
            coefs = coeficientes_dct(matriz, n, log=log)
            for j in range(n):
                novo[f"DCT_{nome}_{j}"] = coefs[:, j]

    if com_geo is None:
        com_geo = USAR_GEOMETRIA_ESPECTRAL
    if com_geo:
        for prefixo, matriz in [("ASE", ase), ("SFM", sfm)]:
            for nome, valores in _geometria_espectral(matriz, prefixo).items():
                novo[nome] = valores

    return novo


def ajustar_tratamento(X_tr, skew_limite=2.0, quantis=(0.005, 0.995)):
    skew = X_tr.skew()
    cols = skew[skew.abs() > skew_limite].index.tolist()
    X_log = X_tr.copy()
    for c in cols:
        X_log[c] = np.sign(X_log[c]) * np.log1p(np.abs(X_log[c]))
    return cols, X_log.quantile(quantis[0]), X_log.quantile(quantis[1])


def aplicar_tratamento(X, cols, li, ls):
    X = X.copy()
    for c in cols:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X.clip(li, ls, axis=1)


# pasta padrão: <raiz-do-repo>/data (script vive em <raiz>/solucao_final/)
PASTA_DADOS_PADRAO = Path(__file__).resolve().parent.parent / "data"


def carregar(pasta=None):
    pasta = Path(pasta) if pasta is not None else PASTA_DADOS_PADRAO
    tr = pd.read_csv(pasta / "genresTrain.csv")
    te = pd.read_csv(pasta / "genresTest.csv")
    return tr, te

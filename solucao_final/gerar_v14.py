# -*- coding: utf-8 -*-
"""v14 = 4ª passada: receita RECORDE da v13 (LB 0,82386) com UMA mudança
validada (experimento_fracoes_v14.py): fração-alvo estendida de 0,85 -> 0,90.
A validação mostrou que os rótulos incrementais até 90% ainda acertam ~80-84%
(entram com peso ~0,5); de 92% em diante a qualidade colapsa (61-79%) — por
isso o teto fica em 0,90. Professor = probs finais convergidas da v13.

EXECUÇÃO PARALELA:
    python gerar_v14.py 0..7    (um processo por membro, simultâneos)
    python gerar_v14.py final   (agrega os 8 estados -> resultados/submissao_v14.csv)

Dependências geradas pela receita da v13 (não incluídas neste repositório,
que documenta apenas a v14 — ver README na raiz para a receita completa):
    artefatos/clusters_v13.pkl        (grupos híbridos kNN-mútuo, gerar_clusters_v13.py)
    artefatos/estado_v13_m{0..7}.pkl  (probabilidades convergidas de cada membro da v13)
Rode a v13 primeiro (ou copie esses arquivos para artefatos/) antes deste script."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
ARTEFATOS = RAIZ / "artefatos"
RESULTADOS = RAIZ / "resultados"

from pipeline_comum import carregar, criar_features, ajustar_tratamento, aplicar_tratamento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

# 2ª passada: menos rodadas (o ponto de partida já está convergido) e frações
# altas desde o início — a iteração refina a composição do pool, não a expande
FRACOES_IT = [0.85, 0.87, 0.89, 0.90, 0.90, 0.90]   # teto 0,90 (validado; 0,92+ colapsa)
FRAC_SEMENTE = 0.85     # fração do teste semeada na rodada 0 com as probs da v13
MEMBROS = [(42, 10, 63, 1.00, FRACOES_IT),
           (42, 10, 63, 0.90, FRACOES_IT),
           (7, 12, 47, 0.90, FRACOES_IT),
           (2026, 8, 95, 0.90, FRACOES_IT),
           (777, 11, 79, 0.90, FRACOES_IT),
           (1234, 9, 55, 0.90, FRACOES_IT),
           (555, 10, 47, 0.90, FRACOES_IT),
           (99, 12, 95, 0.90, FRACOES_IT)]
CONF_MINIMA, PESO_BASE, PRIOR_MINIMO = 0.70, 0.8, 0.005
PESO_CONSENSO = 0.9     # validado: rótulos de consenso são mais precisos -> peso maior
PESOS_ENSEMBLE = (0.5, 0.3, 0.2)
LIMIAR_CONSENSO = 0.60    # conf. média mínima do cluster p/ rótulo por consenso
                          # (0,80 quase não resgatava: clusters discordantes raramente
                          #  fecham consenso alto; 0,60 = pluralidade clara, e o
                          #  RE-TREINO das rodadas seguintes corrige os erros que entrarem)
MIN_COLEGAS = 2           # nº mínimo de membros de teste no cluster
COD = {"Blues": 1, "Classical": 2, "Jazz": 3, "Metal": 4, "Pop": 5, "Rock": 6}

df_tr, df_te = carregar()
y_full = df_tr["GENRE"].values
n_tr, n_te = len(df_tr), len(df_te)
X_fe_full = criar_features(df_tr.drop(columns=["GENRE"]), com_dct=True, com_geo=False)
X_te_fe = criar_features(df_te, com_dct=True, com_geo=False)
print(f"Features (idênticas à v6): {X_fe_full.shape[1]}", flush=True)

# grupos HÍBRIDOS (kNN-mútuo k=3 com fallback para o fino) — validados nas etapas 1/2
rot_hib = joblib.load(ARTEFATOS / "clusters_v13.pkl")["rot"]
cl_te = rot_hib[n_tr:]                        # grupo de cada segmento de teste
K = rot_hib.max() + 1
n_te_cl = np.bincount(cl_te, minlength=K)     # nº de colegas de teste por grupo
print(f"Grupos híbridos: {K} | segmentos de teste com {MIN_COLEGAS}+ colegas de teste: "
      f"{(n_te_cl[cl_te] >= MIN_COLEGAS).mean():.1%}", flush=True)

# probabilidades finais convergidas da v13 (recorde 0,82386): o "professor" da 4ª passada
membros_v13 = [joblib.load(ARTEFATOS / f"estado_v13_m{i}.pkl") for i in range(8)]
CLASSES_V9 = membros_v13[0]["classes"]
PROB_V9 = np.mean([m["prob"] for m in membros_v13], axis=0)
print(f"Semente: probs da v13 carregadas (conf média {PROB_V9.max(1).mean():.3f})", flush=True)


def corrigir_label_shift(prob, prior_treino):
    prior = prior_treino.copy()
    for _ in range(30):
        prob_aj = prob * (prior / prior_treino)
        prob_aj /= prob_aj.sum(axis=1, keepdims=True)
        prior_novo = np.clip(prob_aj.mean(axis=0), PRIOR_MINIMO, None)
        prior_novo /= prior_novo.sum()
        if np.abs(prior_novo - prior).max() < 1e-6:
            break
        prior = prior_novo
    return prob_aj, prior


def executar_membro(im, seed, c_svm, folhas, frac_bag, fracoes):
    if frac_bag < 1.0:
        idx_bag, _ = train_test_split(np.arange(n_tr), train_size=frac_bag,
                                      stratify=y_full, random_state=seed)
    else:
        idx_bag = np.arange(n_tr)
    X_bag, y = X_fe_full.iloc[idx_bag], y_full[idx_bag]

    cols, li, ls = ajustar_tratamento(X_bag)
    A = aplicar_tratamento(X_bag, cols, li, ls)
    B = aplicar_tratamento(X_te_fe, cols, li, ls)
    sc = StandardScaler()
    A, B = sc.fit_transform(A), sc.transform(B)

    # ---- rodada 0 SEMEADA com as probs convergidas da v9 (2ª passada do laço) ----
    lab9, conf9 = PROB_V9.argmax(1), PROB_V9.max(1)
    prior9 = PROB_V9.mean(axis=0)
    n_alvo0 = FRAC_SEMENTE * n_te
    sel0 = np.concatenate([
        (ic := np.where((lab9 == i) & (conf9 >= LIMIAR_CONSENSO))[0])
        [np.argsort(-conf9[ic])[:int(round(prior9[i] * n_alvo0))]]
        for i in range(len(CLASSES_V9))])
    X_atual = np.vstack([A, B[sel0]])
    y_atual = np.r_[y, CLASSES_V9[lab9[sel0]]]
    w_atual = np.r_[np.ones(len(y)), PESO_BASE * conf9[sel0]]
    print(f"[m{im}] semente v9: {len(sel0)} pseudo-rótulos iniciais", flush=True)

    historico, pred_ant = [], None
    for r, fracao in enumerate(fracoes):
        t0 = time.time()
        svm = SVC(C=c_svm, gamma="scale", probability=True, random_state=seed,
                  cache_size=2000).fit(X_atual, y_atual, sample_weight=w_atual)
        hgb = HistGradientBoostingClassifier(max_iter=300, random_state=seed
                  ).fit(X_atual, y_atual, sample_weight=w_atual)
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=folhas,
                             random_state=seed, n_jobs=-1, verbose=-1
                  ).fit(X_atual, y_atual, sample_weight=w_atual)
        classes = svm.classes_
        assert list(classes) == list(CLASSES_V9)
        a, b, c = PESOS_ENSEMBLE
        p_lgb = lgb.predict_proba(B)[:, [list(lgb.classes_).index(cl) for cl in classes]]
        prob = a * svm.predict_proba(B) + b * hgb.predict_proba(B) + c * p_lgb

        pt = pd.Series(w_atual).groupby(pd.Series(y_atual)).sum()
        pt = (pt / pt.sum()).reindex(classes).values
        prob_aj, prior_est = corrigir_label_shift(prob, pt)
        pred_i, conf_i = prob_aj.argmax(1), prob_aj.max(1)
        historico = (historico + [prob_aj])[-3:]

        # ---------- CONSENSO DE MÚSICA (só colegas de teste do cluster fino) ----------
        S = np.zeros((K, len(classes)))
        np.add.at(S, cl_te, prob_aj)
        media_cl = S[cl_te] / n_te_cl[cl_te, None]          # prob média do cluster
        cons_lab, cons_conf = media_cl.argmax(1), media_cl.max(1)
        usa_cons = (n_te_cl[cl_te] >= MIN_COLEGAS) & (cons_conf >= LIMIAR_CONSENSO)

        lab_cand = np.where(usa_cons, cons_lab, pred_i)      # rótulo candidato
        conf_cand = np.where(usa_cons, cons_conf, conf_i)    # confiança efetiva
        resgatados = usa_cons & (cons_lab != pred_i)         # consenso venceu o individual

        # seleção com cotas por classe ∝ prior estimado (inalterada);
        # piso: consenso usa o próprio limiar, individual usa CONF_MINIMA
        piso = np.where(usa_cons, LIMIAR_CONSENSO, CONF_MINIMA)
        n_alvo = fracao * n_te
        sel = np.concatenate([
            (ic := np.where((lab_cand == i) & (conf_cand >= piso))[0])
            [np.argsort(-conf_cand[ic])[:int(round(prior_est[i] * n_alvo))]]
            for i in range(len(classes))])

        pred = classes[lab_cand]
        mud = "-" if pred_ant is None else str(int((pred != pred_ant).sum()))
        print(f"[m{im} seed {seed}] rodada {r}: aceitos = {len(sel):>4} "
              f"(consenso: {int(usa_cons[sel].sum())}, resgatados: {int(resgatados[sel].sum())}) | "
              f"mudaram = {mud:>4} | conf = {conf_i.mean():.3f} | "
              f"Metal = {prior_est[list(classes).index('Metal')]:.1%} | {time.time()-t0:5.1f}s",
              flush=True)
        pred_ant = pred.copy()
        X_atual = np.vstack([A, B[sel]])
        y_atual = np.r_[y, classes[lab_cand[sel]]]
        # peso maior para entradas por consenso (rótulos comprovadamente mais precisos)
        peso_sel = np.where(usa_cons[sel], PESO_CONSENSO, PESO_BASE)
        w_atual = np.r_[np.ones(len(y)), peso_sel * conf_cand[sel]]

    media = np.mean(historico, axis=0)
    ARTEFATOS.mkdir(exist_ok=True)
    joblib.dump({"prob": media, "classes": classes}, ARTEFATOS / f"estado_v14_m{im}.pkl")
    return media, classes


MODO = sys.argv[1] if len(sys.argv) > 1 else "final"

if MODO != "final":
    im = int(MODO)
    seed, c_svm, folhas, frac, fracoes = MEMBROS[im]
    print(f"===== membro {im}: seed={seed} C={c_svm} folhas={folhas} frac={frac} "
          f"({len(fracoes)} rodadas) =====", flush=True)
    executar_membro(im, seed, c_svm, folhas, frac, fracoes)
    print(f"[m{im}] estado salvo.", flush=True)
    sys.exit(0)

# ---------------------- agregação final ----------------------
probs, classes_ref = [], None
for im in range(len(MEMBROS)):
    est = joblib.load(ARTEFATOS / f"estado_v14_m{im}.pkl")
    if classes_ref is None:
        classes_ref = est["classes"]
    assert list(est["classes"]) == list(classes_ref)
    probs.append(est["prob"])

prob_final = np.mean(probs, axis=0)
pred_final = np.asarray(classes_ref)[prob_final.argmax(1)]
RESULTADOS.mkdir(exist_ok=True)
caminho_sub = RESULTADOS / "submissao_v14.csv"
sub = pd.DataFrame({"Id": np.arange(1, n_te + 1), "Genres": [COD[g] for g in pred_final]})
sub.to_csv(caminho_sub, index=False)

inv = {v: k for k, v in COD.items()}
print(f"\n'{caminho_sub.name}' salva em {caminho_sub.parent}.")

caminho_v13 = RESULTADOS / "submissao_v13.csv"
if caminho_v13.exists():
    ref = pd.read_csv(caminho_v13)
    dif = int((sub["Genres"].values != ref["Genres"].values).sum())
    print(f"Difere da v13 (0,82386) em {dif} previsões ({dif/n_te:.1%})")
else:
    print("submissao_v13.csv não encontrada em resultados/ — comparação com a v13 pulada "
          "(este repositório documenta apenas a v14).")

preds_m = [np.asarray(classes_ref)[p.argmax(1)] for p in probs]
unanime = np.mean([len({p[i] for p in preds_m}) == 1 for i in range(n_te)])
print(f"Unanimidade entre os {len(MEMBROS)} membros: {unanime:.1%}")
print("Distribuição: " + " | ".join(f"{inv[k]}: {v}" for k, v in sub["Genres"].value_counts().items()))

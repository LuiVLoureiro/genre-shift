# solucao_final/ — scripts da v14

- `pipeline_comum.py` — features (FE_*, DCT_*) e tratamento de outliers, compartilhados por todas as versões.
- `gerar_v14.py` — receita completa da v14 (recorde, LB 0,83024). `python gerar_v14.py 0..7` roda um membro do bagging; `python gerar_v14.py final` agrega os 8 em `resultados/submissao_v14.csv`.
- `experimento_fracoes_v14.py` — experimento de validação que aprovou a mudança da v14 (fração-alvo 0,85 → 0,90) antes de gastar uma submissão.
- `orquestrador_v14.py` — roda os 8 membros em lotes e a agregação final sem intervenção manual.

Dependências: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `joblib`
(ver `requirements.txt` na raiz).

`gerar_v14.py` parte dos artefatos convergidos da v13 (ver `artefatos/README.md`)
— não incluídos neste repositório, que documenta apenas a v14.

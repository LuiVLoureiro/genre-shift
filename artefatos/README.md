# Artefatos intermediários

Pasta de saída/entrada dos estados intermediários usados por
`solucao_final/gerar_v14.py`:

- `clusters_v13.pkl` — grupos híbridos kNN-mútuo (gerados pela receita da v13)
- `estado_v13_m{0..7}.pkl` — probabilidades convergidas de cada membro da v13,
  usadas como "professor"/semente da v14
- `estado_v14_m{0..7}.pkl` — gerados por `gerar_v14.py <0..7>`, consumidos por
  `gerar_v14.py final`

Este repositório documenta apenas a v14; os artefatos da v13 listados acima
não estão incluídos aqui (foram gerados em execução local e não persistidos).
Para reproduzir a v14 do zero, rode a receita da v13 primeiro para gerar esses
três primeiros arquivos nesta pasta.

Os arquivos `.pkl` gerados localmente não são versionados (ver `.gitignore`).

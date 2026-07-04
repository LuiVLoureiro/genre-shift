# Dados

Este repositório documenta a solução, mas não redistribui o dataset da
competição (`genresTrain.csv`, `genresTest.csv`) — ele pertence à competição
Kaggle da disciplina de Fundamentos de Inteligência Artificial (Mestrado,
UFPA), criada pelo professor.

Para reproduzir os scripts em `solucao_final/`, baixe os dois arquivos da
página da competição no Kaggle e coloque-os nesta pasta:

```
data/
├── genresTrain.csv
└── genresTest.csv
```

`solucao_final/pipeline_comum.py::carregar()` lê os arquivos desta pasta por
padrão.

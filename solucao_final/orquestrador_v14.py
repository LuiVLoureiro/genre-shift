# -*- coding: utf-8 -*-
"""Orquestrador autônomo da v14 — completa tudo sem intervenção:
1. espera os estados do lote 1 (membros 0-3, já em execução);
2. dispara o lote 2 (membros 4-7) em paralelo e espera;
3. roda a agregação final -> resultados/submissao_v14.csv.
Timeout de segurança: 4h por fase."""
import sys, time, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent / "gerar_v14.py"
ARTEFATOS = RAIZ / "artefatos"
TIMEOUT = 4 * 3600


def esperar_estados(indices, fase):
    t0 = time.time()
    faltam = set(indices)
    while faltam and time.time() - t0 < TIMEOUT:
        prontos = {i for i in faltam if (ARTEFATOS / f"estado_v14_m{i}.pkl").exists()}
        if prontos:
            faltam -= prontos
            print(f"[{fase}] estados prontos: {sorted(set(indices) - faltam)} "
                  f"(faltam {sorted(faltam)})", flush=True)
        if faltam:
            time.sleep(60)
    if faltam:
        raise TimeoutError(f"[{fase}] timeout esperando membros {sorted(faltam)}")


print("Orquestrador v14 iniciado. Fase 1: aguardando lote 1 (membros 0-3)...", flush=True)
esperar_estados([0, 1, 2, 3], "lote 1")

print("Fase 2: lançando lote 2 (membros 4-7) em paralelo...", flush=True)
procs = []
for im in [4, 5, 6, 7]:
    log = open(RAIZ / f"v14_m{im}.log", "w", encoding="utf-8")
    procs.append((im, subprocess.Popen([sys.executable, str(SCRIPT), str(im)],
                                       stdout=log, stderr=subprocess.STDOUT), log))
for im, p, log in procs:
    codigo = p.wait(timeout=TIMEOUT)
    log.close()
    print(f"  membro {im}: exit code {codigo}", flush=True)
    if codigo != 0:
        raise RuntimeError(f"membro {im} falhou — ver v14_m{im}.log")

print("Fase 3: agregação final...", flush=True)
r = subprocess.run([sys.executable, str(SCRIPT), "final"], capture_output=True,
                   text=True, encoding="utf-8", timeout=1800)
print(r.stdout, flush=True)
if r.returncode != 0:
    print(r.stderr, flush=True)
    raise RuntimeError("agregação final falhou")
print("\n=== ORQUESTRADOR CONCLUÍDO: resultados/submissao_v14.csv pronta. ===", flush=True)

"""Demo de ponta a ponta, so com dados sinteticos e sem banco:

    python -m detector_anomalias.demo
    python -m detector_anomalias.demo --semente 7 --simulacoes 5 --saida relatorio.html

Gera as transferencias, treina no passado, analisa o ultimo mes, compara com a
regra simples e grava um relatorio HTML. A semente da demo e as das simulacoes sao
separadas das usadas para ajustar o modelo (100 a 109), que nao entram no resultado.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from detector_anomalias.avaliacao import avaliar_conjunto, avaliar_varias_sementes, resumir
from detector_anomalias.relatorio import DadosRelatorio, gerar_html
from detector_anomalias.sintetico import gerar

SAIDA_PADRAO = Path("artefatos") / "anomalias" / "relatorio.html"
PRIMEIRA_SEMENTE_FINAL = 300


def executar(semente: int, simulacoes: int, saida: Path) -> DadosRelatorio:
    conjunto = gerar(semente)
    avaliacao, detector, teste = avaliar_conjunto(conjunto, semente=semente)
    resultados = detector.analisar(teste)
    assert detector.limiar is not None
    avaliacoes = avaliar_varias_sementes(
        range(PRIMEIRA_SEMENTE_FINAL, PRIMEIRA_SEMENTE_FINAL + simulacoes)
    )
    dados = DadosRelatorio(conjunto, resultados, detector.limiar, avaliacoes)

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(gerar_html(dados), encoding="utf-8")

    print("DADOS 100% SINTETICOS")
    print(f"Treino: {avaliacao.treino} notas. Analisadas: {avaliacao.teste} notas novas.")
    sinalizadas = sorted(
        (r for r in resultados if r.sinalizada), key=lambda r: r.score, reverse=True
    )
    print(f"\nSinalizadas: {len(sinalizadas)}. As 5 mais fora do padrao:")
    for r in sinalizadas[:5]:
        print(f"  {r.score:.2f}  {r.registro.id}  {r.registro.origem} -> {r.registro.destino}")
        for motivo in r.motivos:
            print(f"        - {motivo.texto}")

    print()
    for nome, qual in (("Modelo", "modelo"), ("Regra simples", "regra_simples")):
        recall = resumir([getattr(a, qual).recall for a in avaliacoes])
        precisao = resumir([getattr(a, qual).precisao for a in avaliacoes])
        print(
            f"{nome}, {len(avaliacoes)} simulacoes: "
            f"recall {recall.media:.2f} ± {recall.desvio:.2f}, "
            f"precisao {precisao.media:.2f} ± {precisao.desvio:.2f}"
        )
    print(f"\nRelatorio: {saida.resolve()}")
    return dados


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo do detector de anomalias (dados sinteticos)")
    parser.add_argument("--semente", type=int, default=42, help="semente do conjunto exibido")
    parser.add_argument("--simulacoes", type=int, default=20, help="simulacoes na comparacao")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO, help="arquivo HTML de saida")
    args = parser.parse_args(argv)
    if args.simulacoes < 1:
        parser.error("--simulacoes precisa ser pelo menos 1")
    # O console do Windows nem sempre usa UTF-8 e quebraria nos acentos dos motivos.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    executar(args.semente, args.simulacoes, args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())

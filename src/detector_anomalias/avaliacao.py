"""Mede o detector contra as anomalias injetadas e contra uma regra simples.

Corte temporal, como na vida real: treina no passado e mede no futuro. Os rotulos
so entram aqui, nunca no treino.

Comparacao justa: o modelo e a regra simples conferem a MESMA quantidade de notas
(as `k` de maior score), porque o limite real e o tempo de quem confere. A regra
simples da a cada nota o seu pior percentil entre as features, como faria quem
olha cada coluna separada. Se o modelo nao ganhar dela, ele nao justifica o custo.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from detector_anomalias.features import ExtratorFeatures, Matriz
from detector_anomalias.modelo import DetectorAnomalias
from detector_anomalias.registros import RegistroTransferencia
from detector_anomalias.sintetico import TIPOS_ANOMALIA, ConjuntoSintetico, TipoAnomalia, gerar


@dataclass(frozen=True)
class Metricas:
    """Desempenho ao conferir as `k` notas de maior score."""

    k: int
    acertos: int
    precisao: float
    recall: float
    precisao_media: float
    recall_por_tipo: dict[TipoAnomalia, float]


@dataclass(frozen=True)
class Avaliacao:
    semente: int
    treino: int
    teste: int
    anomalias_teste: int
    modelo: Metricas
    regra_simples: Metricas


@dataclass(frozen=True)
class Resumo:
    """Media e desvio padrao de uma metrica entre as sementes."""

    media: float
    desvio: float


def separar_por_tempo(
    registros: Sequence[RegistroTransferencia], fracao_treino: float = 5 / 6
) -> tuple[list[RegistroTransferencia], list[RegistroTransferencia]]:
    """Primeiros registros no treino, ultimos no teste. Exige ordem de emissao."""
    corte = int(len(registros) * fracao_treino)
    return list(registros[:corte]), list(registros[corte:])


def scores_regra_simples(treino: Matriz, teste: Matriz) -> Matriz:
    """O pior percentil de cada nota entre as features, com referencia no treino.

    Percentil, e nao desvio padrao: as features tem escalas muito diferentes, e um
    desvio padrao favoreceria as quase constantes. Empate no pior percentil (nota
    alem do maximo do treino em mais de uma feature) sai pela media dos percentis.
    """
    centro = np.median(treino, axis=0)
    referencia = np.sort(np.abs(treino - centro), axis=0)
    distancias = np.abs(teste - centro)
    percentis = np.column_stack(
        [
            np.searchsorted(referencia[:, j], distancias[:, j], side="right") / len(treino)
            for j in range(treino.shape[1])
        ]
    )
    return np.asarray(percentis.max(axis=1) + 1e-3 * percentis.mean(axis=1), dtype=np.float64)


def medir(
    scores: Matriz,
    registros: Sequence[RegistroTransferencia],
    rotulos: dict[str, TipoAnomalia],
    k: int,
) -> Metricas:
    if len(scores) != len(registros):
        raise ValueError("scores e registros com tamanhos diferentes")
    positivos = np.array([r.id in rotulos for r in registros])
    ordem = np.argsort(-scores, kind="stable")[:k]
    conferidos = {registros[int(i)].id for i in ordem}
    acertos = int(positivos[ordem].sum())
    total = int(positivos.sum())

    por_tipo: dict[TipoAnomalia, float] = {}
    for tipo in TIPOS_ANOMALIA:
        ids = [r.id for r in registros if rotulos.get(r.id) == tipo]
        if ids:
            por_tipo[tipo] = sum(i in conferidos for i in ids) / len(ids)

    return Metricas(
        k=k,
        acertos=acertos,
        precisao=acertos / k if k else 0.0,
        recall=acertos / total if total else 0.0,
        precisao_media=float(average_precision_score(positivos, scores)) if total else 0.0,
        recall_por_tipo=por_tipo,
    )


def avaliar_conjunto(
    conjunto: ConjuntoSintetico, *, semente: int = 42, fracao_alerta: float = 0.02
) -> tuple[Avaliacao, DetectorAnomalias, list[RegistroTransferencia]]:
    """Treina no passado e mede no futuro, com o modelo e com a regra simples."""
    treino, teste = separar_por_tempo(conjunto.registros)
    detector = DetectorAnomalias(fracao_alerta=fracao_alerta, semente=semente).treinar(treino)
    k = max(1, round(len(teste) * fracao_alerta))

    extrator = ExtratorFeatures().ajustar(treino)
    simples = scores_regra_simples(extrator.transformar(treino), extrator.transformar(teste))
    avaliacao = Avaliacao(
        semente=semente,
        treino=len(treino),
        teste=len(teste),
        anomalias_teste=sum(r.id in conjunto.rotulos for r in teste),
        modelo=medir(detector.pontuar(teste), teste, conjunto.rotulos, k),
        regra_simples=medir(simples, teste, conjunto.rotulos, k),
    )
    return avaliacao, detector, teste


def avaliar_varias_sementes(
    sementes: Sequence[int], *, quantidade: int = 5_000, fracao_alerta: float = 0.02
) -> list[Avaliacao]:
    """Uma so semente pode dar sorte: cada uma gera outras empresas, rotas e anomalias."""
    return [
        avaliar_conjunto(gerar(s, quantidade=quantidade), semente=s, fracao_alerta=fracao_alerta)[0]
        for s in sementes
    ]


def resumir(valores: Sequence[float]) -> Resumo:
    arr = np.asarray(valores, dtype=np.float64)
    return Resumo(float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

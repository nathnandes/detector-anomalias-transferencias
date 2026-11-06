"""O formato de entrada do detector."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RegistroTransferencia:
    """Uma transferencia, com o que a nota de saida e a de entrada informam.

    `emitida_em` e `recebida_em` vem no horario local das empresas: a hora do dia
    e uma das features. `recebida_em` e `quantidade_recebida` ficam nulos enquanto
    a entrada no destino nao acontece. As quantidades sao a soma dos itens.
    """

    id: str
    origem: str
    destino: str
    valor: float
    emitida_em: datetime
    recebida_em: datetime | None
    quantidade_itens: int
    quantidade_enviada: float
    quantidade_recebida: float | None

    @property
    def rota(self) -> tuple[str, str]:
        return (self.origem, self.destino)

    @property
    def pendente(self) -> bool:
        return self.recebida_em is None

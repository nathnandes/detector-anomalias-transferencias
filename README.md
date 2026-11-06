# Detector de anomalias em transferências

Quando uma rede de empresas transfere mercadoria entre si todos os dias, ninguém consegue conferir
nota por nota. Este projeto aprende o padrão de cada rota, aponta as notas que fogem dele e **diz o
motivo em português**, para quem confere saber onde olhar:

```text
0.65  SIM-000290  Empresa 05 -> Empresa 06
      - valor de R$ 32.715,42, 5,3x a mediana da rota (R$ 6.137,36)
      - emitida às 19h, horário de 1,8% das notas
      - R$ 2.336,82 por item, 3,6x o normal da rota
```

> **Todos os dados aqui são inventados.** Empresas, valores e notas saem de um gerador sintético.
> Nenhum dado real de empresa está neste repositório.

## Rodar

Precisa de Python 3.12. Não precisa de banco nem de rede.

```bash
python -m venv .venv
source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m detector_anomalias.demo
```

Em menos de um minuto, o comando mostra as notas sinalizadas e grava um relatório em
`artefatos/anomalias/relatorio.html`. Há um exemplo pronto em
[exemplo/relatorio.html](exemplo/relatorio.html): baixe e abra no navegador.

Para rodar os testes: `pytest`.

## Como funciona

**Por que não supervisionado.** Não existe um histórico confiável de "isto foi fraude" ou "isto foi
erro" para treinar um classificador. O [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
parte os dados com cortes aleatórios: o que fica isolado com poucos cortes é diferente do resto.

| Etapa | Arquivo | O que faz |
|---|---|---|
| Dados | [`sintetico.py`](src/detector_anomalias/sintetico.py) | Gera 5.000 transferências em 180 dias e injeta anomalias conhecidas em 1,5% delas |
| Características | [`features.py`](src/detector_anomalias/features.py) | Aprende o padrão de cada rota e mede o desvio de cada nota |
| Modelo | [`modelo.py`](src/detector_anomalias/modelo.py) | Treina, define o limiar de alerta e explica o motivo |
| Avaliação | [`avaliacao.py`](src/detector_anomalias/avaliacao.py) | Treina no passado, mede no futuro e compara com uma regra simples |

Quatro decisões que importam:

- **Cada nota é comparada com a própria rota.** R$ 5.000 é normal numa rota e absurdo em outra. Rota
  com pouco histórico é puxada para o padrão geral, para não julgar com base em duas notas.
- **O limiar vem da capacidade de conferir.** O padrão é 2% das notas. Não é uma estimativa de
  quantas têm problema, porque esse número ninguém conhece.
- **O motivo vem do próprio modelo.** Cada característica volta ao valor típico e o score é
  recalculado: o que mais derruba o score é o motivo principal. É uma aproximação, porque mede uma
  característica por vez e não as interações.
- **O futuro não vaza para o passado.** O padrão de cada rota e o modelo usam só o período de treino.

## Resultados

São 20 simulações, cada uma com outras rotas, valores e anomalias, treinando em 4.166 notas e
analisando as 834 seguintes. O modelo e a regra simples conferem as mesmas 17 notas (2%). A regra
simples dá a cada nota o seu pior percentil entre as características, como faria alguém olhando
uma coluna por vez.

| Média ± desvio | Precisão | Recall | Precisão média (AP) |
|---|---|---|---|
| Modelo (IsolationForest) | 0,57 ± 0,18 | 0,83 ± 0,11 | 0,84 ± 0,10 |
| Regra simples | 0,35 ± 0,14 | 0,52 ± 0,16 | 0,52 ± 0,17 |

| Parte encontrada de cada tipo | Modelo | Regra simples |
|---|---|---|
| Valor muito alto | 84% | 43% |
| Fora de horário | 100% | 73% |
| Rota rara | 100% | 75% |
| Trânsito longo | 24% | 19% |
| Divergência de quantidade | 100% | 31% |
| Combinada sutil | 86% | 63% |

A precisão tem um teto: há cerca de 12 anomalias nas 834 notas e 17 conferências, então nem um
modelo perfeito chegaria a 100%.

## O que estes números não dizem

- **Não medem a eficácia com dados reais.** O gerador foi escrito junto com o detector, então os
  números mostram que o método funciona nestes padrões. O resultado real só aparece com pessoas
  conferindo as notas sinalizadas.
- **Trânsito longo é o ponto fraco.** Uma nota que só destoa no tempo até a entrada, mesmo com 5 a
  10 vezes o normal, fica diluída entre as outras características. Esse caso pede uma regra de prazo
  por rota, não mais modelo.
- **O modelo não enxerga uma característica que nunca variou.** Se nenhuma nota do histórico tiver
  divergência acima da tolerância, a coluna fica constante e uma falta de 80% passa sem sinal.
- **Fora do padrão não significa erro nem fraude.** É um pedido de conferência. Por isso o modelo não
  usa quem emitiu a nota: ele não aponta para uma pessoa.

### Ajustes feitos olhando os resultados

| Mudança | Motivo |
|---|---|
| A regra simples passou de desvio padrão para percentil | A primeira versão favorecia o modelo: qualquer sábado parecia extremo |
| A anomalia "sutil" passou a usar horários que também ocorrem em notas normais | Antes, a hora sozinha a denunciava |
| A divergência ganhou uma tolerância aprendida | Qualquer quebra pequena parecia grave |
| A quantidade de itens saiu das características | Repetia o que valor e valor por item já dizem |
| A raridade só conta abaixo de 5% das notas | Sábado é menos comum, mas é rotina |

Esses ajustes usaram as sementes 100 a 109. As tabelas acima usam as sementes 300 a 319, que não
participaram de nenhum deles.

## Origem do projeto

O detector nasceu como parte de um sistema interno de monitoramento de transferências entre 6
empresas, onde roda em um worker próprio, grava os scores e alimenta um painel e um alerta. Esse
sistema é privado e não está aqui. Até agora, o detector só foi testado com dados sintéticos.

Feito com Python, scikit-learn e NumPy. Desenvolvido com apoio do Claude Code.

"""Cadeia de Markov de ordem n configurável, genérica para qualquer símbolo
"hashable" (tuplas, strings, números).

- Símbolos especiais INICIO e FIM delimitam cada sequência do corpus.
- Probabilidades de transição = contagens normalizadas (máxima verossimilhança).
- Backoff: se o contexto de ordem n nunca foi visto (ou todas as continuações
  foram proibidas por uma restrição), o modelo tenta a ordem n-1, n-2, ..., 0.
- Semente aleatória: cada geração usa um random.Random(semente) próprio,
  então a mesma semente reproduz exatamente a mesma sequência.
"""

import random
from collections import Counter, defaultdict

INICIO = "<INICIO>"
FIM = "<FIM>"


class CadeiaMarkov:
    def __init__(self, ordem):
        if ordem < 1:
            raise ValueError("A ordem deve ser >= 1")
        self.ordem = ordem
        # transicoes[k][contexto] = Counter(proximo_simbolo -> contagem),
        # para k = 0..ordem (k = 0 é a distribuição sem contexto).
        self.transicoes = [defaultdict(Counter) for _ in range(ordem + 1)]

    # ------------------------------------------------------------------ treino
    def treinar(self, sequencias):
        """Conta as transições de todas as ordens 0..n em cada sequência."""
        n = self.ordem
        for seq in sequencias:
            s = [INICIO] * n + list(seq) + [FIM]
            for i in range(n, len(s)):
                for k in range(n + 1):
                    contexto = tuple(s[i - k:i])
                    self.transicoes[k][contexto][s[i]] += 1
        return self

    def probabilidades(self, contexto):
        """Distribuição P(próximo | contexto) na ordem máxima (sem backoff)."""
        cont = self.transicoes[self.ordem].get(tuple(contexto[-self.ordem:]))
        if not cont:
            return {}
        total = sum(cont.values())
        return {s: c / total for s, c in cont.items()}

    # ---------------------------------------------------------------- geração
    def proximo(self, historico, rng, permitido=None):
        """Sorteia o próximo símbolo dado o histórico (já com INICIOs).

        permitido(simbolo) -> bool filtra candidatos (ex.: proibir FIM).
        Devolve (simbolo, ordem_usada). ordem_usada < self.ordem indica backoff.
        """
        for k in range(self.ordem, -1, -1):
            contexto = tuple(historico[len(historico) - k:]) if k else ()
            cont = self.transicoes[k].get(contexto)
            if not cont:
                continue
            candidatos = [(s, c) for s, c in cont.items()
                          if permitido is None or permitido(s)]
            if candidatos:
                simbolos, pesos = zip(*candidatos)
                return rng.choices(simbolos, weights=pesos, k=1)[0], k
        return None, -1  # nenhum símbolo permitido em nenhuma ordem

    def gerar(self, semente=None, max_simbolos=1000, permitido=None, parar=None):
        """Gera uma sequência (sem INICIO/FIM).

        permitido(simbolo, gerado) -> bool: restrição aplicada a cada passo.
        parar(gerado) -> bool: encerra a geração antes de sortear FIM
        (usado para parar ao completar os compassos).
        Devolve (sequencia, ordens_usadas).
        """
        rng = random.Random(semente)
        historico = [INICIO] * self.ordem
        gerado, ordens = [], []
        while len(gerado) < max_simbolos:
            if parar is not None and parar(gerado):
                break
            filtro = (lambda s: permitido(s, gerado)) if permitido else None
            simbolo, k = self.proximo(historico, rng, filtro)
            if simbolo is None or simbolo == FIM:
                break
            gerado.append(simbolo)
            ordens.append(k)
            historico.append(simbolo)
        return gerado, ordens

    # -------------------------------------------------------------- análise
    def fator_ramificacao(self):
        """Média de continuações distintas por contexto (ordem máxima) e
        fração de contextos com uma única continuação (determinísticos)."""
        conts = self.transicoes[self.ordem].values()
        tamanhos = [len(c) for c in conts]
        media = sum(tamanhos) / len(tamanhos)
        determ = sum(1 for t in tamanhos if t == 1) / len(tamanhos)
        return media, determ

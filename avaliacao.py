"""Avaliação do trade-off entre ordem baixa (aleatório) e ordem alta (cópia).

Para cada ordem n, gera várias melodias (uma por semente) e mede:
  1. Taxa de cópia: % dos trechos de k símbolos consecutivos da melodia gerada
     que aparecem idênticos em algum reel do corpus.
  2. Maior trecho copiado: tamanho (em símbolos) da maior sequência gerada que
     existe igual no corpus.
  3. Fator de ramificação: número médio de continuações distintas por contexto
     de ordem n visto no treino.
Como os símbolos são intervalos, "cópia" aqui ignora transposição: o mesmo
trecho tocado em outra tonalidade conta como cópia.

Uso:
    python avaliacao.py --ordens 1 2 3 4 5 6 --sementes 30 --compassos 16 --k 8
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from markov import CadeiaMarkov
from musica import carregar_corpus, estatisticas_corpus, gerar_melodia, para_simbolos

COR = "#2a78d6"
TINTA, TINTA_SEC = "#0b0b0b", "#52514e"


class IndiceNgramas:
    """Conjuntos de n-gramas do corpus, criados sob demanda e reaproveitados."""

    def __init__(self, sequencias):
        self.sequencias = sequencias
        self.cache = {}

    def conjunto(self, k):
        if k not in self.cache:
            self.cache[k] = {tuple(s[i:i + k]) for s in self.sequencias
                             for i in range(len(s) - k + 1)}
        return self.cache[k]

    def contem_algum(self, seq, k):
        c = self.conjunto(k)
        return any(tuple(seq[i:i + k]) in c for i in range(len(seq) - k + 1))


def taxa_copia(seq, indice, k):
    grams = [tuple(seq[i:i + k]) for i in range(len(seq) - k + 1)]
    c = indice.conjunto(k)
    return sum(g in c for g in grams) / len(grams) if grams else 0.0


def maior_trecho_copiado(seq, indice):
    """Busca binária: se existe trecho copiado de tamanho L, existe de L-1."""
    lo, hi = 0, len(seq)
    while lo < hi:
        meio = (lo + hi + 1) // 2
        if indice.contem_algum(seq, meio):
            lo = meio
        else:
            hi = meio - 1
    return lo


def grafico(x, medias, desvios, titulo, rotulo_y, caminho, percentual=False):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.errorbar(x, medias, yerr=desvios, color=COR, linewidth=2, marker="o",
                markersize=7, capsize=4, elinewidth=1.2)
    for xi, yi in zip(x, medias):  # rótulo direto em cada ponto (poucos pontos)
        txt = f"{yi:.0%}" if percentual else f"{yi:.1f}"
        ax.annotate(txt, (xi, yi), textcoords="offset points", xytext=(9, 0), va="center",
                    fontsize=8, color=TINTA_SEC)
    ax.set_title(titulo, color=TINTA, fontsize=11, loc="left")
    ax.set_xlabel("Ordem n da cadeia", color=TINTA_SEC)
    ax.set_ylabel(rotulo_y, color=TINTA_SEC)
    ax.set_xticks(x)
    if percentual:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylim(-0.03, 1.05)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#b5b4ae")
    ax.tick_params(colors=TINTA_SEC)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Avalia o trade-off ordem baixa x ordem alta.")
    ap.add_argument("--ordens", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--sementes", type=int, default=30, help="melodias por ordem")
    ap.add_argument("--compassos", type=int, default=16)
    ap.add_argument("--k", type=int, default=8,
                    help="tamanho do trecho na taxa de cópia (use k maior que a maior ordem)")
    ap.add_argument("--pasta", default="graficos")
    args = ap.parse_args()

    musicas = carregar_corpus()
    est = estatisticas_corpus(musicas)
    seqs = [para_simbolos(m["notas"]) for m in musicas]
    indice = IndiceNgramas(seqs)
    pasta = Path(args.pasta)
    pasta.mkdir(exist_ok=True)

    linhas = []
    for n in args.ordens:
        modelo = CadeiaMarkov(n).treinar(seqs)
        copias, trechos, backoffs = [], [], []
        for semente in range(args.sementes):
            sim, ordens = gerar_melodia(modelo, args.compassos, semente,
                                        est["primeira_comum"], est["min"], est["max"])
            copias.append(taxa_copia(sim, indice, args.k))
            trechos.append(maior_trecho_copiado(sim, indice))
            backoffs.append(sum(k < n for k in ordens) / len(ordens))
        ramif, determ = modelo.fator_ramificacao()
        linha = {
            "ordem": n,
            "contextos": len(modelo.transicoes[n]),
            "taxa_copia_media": np.mean(copias), "taxa_copia_desvio": np.std(copias),
            "maior_trecho_medio": np.mean(trechos), "maior_trecho_desvio": np.std(trechos),
            "ramificacao_media": ramif, "contextos_deterministicos": determ,
            "backoff_medio": np.mean(backoffs),
        }
        linhas.append(linha)
        print(f"n={n}: cópia {linha['taxa_copia_media']:.1%} | maior trecho "
              f"{linha['maior_trecho_medio']:.1f} símbolos | ramificação {ramif:.2f} | "
              f"determinísticos {determ:.0%} | backoff {linha['backoff_medio']:.1%}")

    with open(pasta / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        for l in linhas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})

    x = [l["ordem"] for l in linhas]
    grafico(x, [l["taxa_copia_media"] for l in linhas], [l["taxa_copia_desvio"] for l in linhas],
            f"Taxa de cópia (trechos de {args.k} notas presentes no corpus)",
            "Trechos copiados", pasta / "taxa_copia.png", percentual=True)
    grafico(x, [l["maior_trecho_medio"] for l in linhas], [l["maior_trecho_desvio"] for l in linhas],
            "Maior trecho copiado do corpus", "Tamanho (notas)", pasta / "maior_trecho.png")
    grafico(x, [l["ramificacao_media"] for l in linhas], [0] * len(linhas),
            "Fator de ramificação", "Continuações por contexto (média)",
            pasta / "ramificacao.png")
    print(f"Gráficos e tabela salvos em {pasta}/ "
          f"(média e desvio padrão de {args.sementes} sementes por ordem)")


if __name__ == "__main__":
    main()

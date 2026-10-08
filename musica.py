"""Gerador de melodias no estilo dos reels irlandeses com cadeia de Markov.

Corpus: reels da coleção O'Neill's Music of Ireland (1850 Melodies), que vem
no pacote music21 (números 1176 a 1555 do livro, compasso C| = 2/2).

Símbolo da cadeia: (intervalo em semitons em relação à nota anterior, duração).
Pausas viram ("P", duração). A primeira nota de cada música tem intervalo 0.

Uso:
    python musica.py --ordem 3 --compassos 16 --semente 42 --saida minha_musica
"""

import argparse
import json
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from markov import CadeiaMarkov, FIM

PASTA = Path(__file__).resolve().parent
CACHE = PASTA / "corpus_reels.json"
TEMPOS_POR_COMPASSO = Fraction(4)  # C| (2/2) = 4 semínimas por compasso
ARQUIVOS_REELS = ["1176-1275.abc", "1276-1375.abc", "1376-1475.abc", "1476-1555.abc"]


# ======================================================================
# Corpus
# ======================================================================
def _extrair_corpus():
    """Lê os reels do music21 e devolve [{numero, titulo, notas:[(altura|None, dur)]}]."""
    from music21 import converter, corpus, meter

    caminhos = {p.name: p for p in corpus.getCorePaths() if "oneills1850" in str(p)}
    musicas = []
    for nome in ARQUIVOS_REELS:
        print(f"  lendo {nome} ...", file=sys.stderr)
        for score in converter.parse(caminhos[nome]).scores:
            numero = int(score.metadata.number)
            ts = score.recurse().getElementsByClass(meter.TimeSignature).first()
            if not (1176 <= numero <= 1555) or ts is None or ts.ratioString != "2/2":
                continue
            notas = []
            for el in score.stripTies().recurse().notesAndRests:
                dur = Fraction(el.duration.quarterLength).limit_denominator(48)
                if el.duration.isGrace or dur == 0:
                    continue  # ornamentos (grace notes) são ignorados
                if el.isRest:
                    notas.append((None, str(dur)))
                elif el.isChord:
                    notas.append((max(p.midi for p in el.pitches), str(dur)))
                else:
                    notas.append((el.pitch.midi, str(dur)))
            while notas and notas[0][0] is None:  # descarta pausas iniciais
                notas.pop(0)
            if len(notas) >= 16:
                musicas.append({"numero": numero, "titulo": score.metadata.title, "notas": notas})
    return musicas


def carregar_corpus():
    """Carrega o corpus do cache JSON (ou extrai do music21 na primeira vez)."""
    if not CACHE.exists():
        print("Extraindo corpus do music21 (só na primeira execução, ~30 s)...", file=sys.stderr)
        CACHE.write_text(json.dumps(_extrair_corpus(), ensure_ascii=False))
    musicas = json.loads(CACHE.read_text())
    for m in musicas:
        m["notas"] = [(a, Fraction(d)) for a, d in m["notas"]]
    return musicas


def para_simbolos(notas):
    """Notas absolutas -> símbolos de intervalo: ("N", intervalo, dur) ou ("P", dur)."""
    simbolos, anterior = [], None
    for altura, dur in notas:
        if altura is None:
            simbolos.append(("P", dur))
        else:
            simbolos.append(("N", 0 if anterior is None else altura - anterior, dur))
            anterior = altura
    return simbolos


def para_notas(simbolos, nota_inicial):
    """Símbolos de intervalo -> notas absolutas [(altura|None, dur)]."""
    notas, altura = [], nota_inicial
    for s in simbolos:
        if s[0] == "P":
            notas.append((None, s[1]))
        else:
            altura += s[1]
            notas.append((altura, s[2]))
    return notas


def estatisticas_corpus(musicas):
    alturas = [a for m in musicas for a, _ in m["notas"] if a is not None]
    primeira = Counter(m["notas"][0][0] for m in musicas).most_common(1)[0][0]
    return {"min": min(alturas), "max": max(alturas), "primeira_comum": primeira}


# ======================================================================
# Geração com restrição de compassos
# ======================================================================
def duracao(s):
    return s[1] if s[0] == "P" else s[2]


def gerar_aabb(modelo, semente, nota_inicial, nota_min, nota_max, compassos_parte=8):
    partes = []
    for i in range(2):  # 0 = parte A, 1 = parte B
        simbolos, _ = gerar_melodia(modelo, compassos_parte, semente + i,
                                    nota_inicial, nota_min, nota_max)
        notas = para_notas(simbolos, nota_inicial)
        partes.append(terminar_na_tonica(ajustar_escala_re_maior(notas)))
    a, b = partes
    return a + a + b + b


def gerar_melodia(modelo, compassos, semente, nota_inicial, nota_min, nota_max):
    """Gera símbolos até completar exatamente `compassos` compassos.

    Restrições aplicadas a cada sorteio (os candidatos proibidos são removidos
    e o sorteio é refeito entre os restantes, com backoff se necessário):
      - FIM é proibido (a música só termina ao completar os compassos);
      - a nota não pode ultrapassar o tempo restante;
      - a altura resultante deve ficar entre nota_min e nota_max.
    """
    alvo = TEMPOS_POR_COMPASSO * compassos

    memo = {}

    def estado(gerado):  # (tempo acumulado, altura atual), calculado 1x por passo
        if len(gerado) not in memo:
            tempo = sum((duracao(s) for s in gerado), Fraction(0))
            altura = nota_inicial + sum(s[1] for s in gerado if s[0] == "N")
            memo[len(gerado)] = (tempo, altura)
        return memo[len(gerado)]

    def permitido(s, gerado):
        if s == FIM:
            return False
        tempo, altura = estado(gerado)
        if duracao(s) > alvo - tempo:
            return False
        return s[0] == "P" or nota_min <= altura + s[1] <= nota_max

    def parar(gerado):
        return estado(gerado)[0] >= alvo

    simbolos, ordens = modelo.gerar(semente=semente, permitido=permitido, parar=parar)
    falta = alvo - estado(simbolos)[0]
    if falta > 0 and simbolos:  # caso raro: nenhuma duração cabia; estende a última nota
        s = simbolos[-1]
        simbolos[-1] = (s[0], s[1] + falta) if s[0] == "P" else (s[0], s[1], s[2] + falta)
    return simbolos, ordens

ESCALA_RE_MAIOR = {2, 4, 6, 7, 9, 11, 1}  # Ré, Mi, Fá#, Sol, Lá, Si, Dó#

def ajustar_escala_re_maior(notas):
    ajustadas = []
    for altura, dur in notas:
        if altura is not None and altura % 12 not in ESCALA_RE_MAIOR:
            altura += 1
        ajustadas.append((altura, dur))
    return ajustadas

def terminar_na_tonica(notas, tonica=2):
    """Troca a última nota (ignorando pausas) pelo Ré mais próximo."""
    notas = list(notas)
    for i in range(len(notas) - 1, -1, -1):
        altura, dur = notas[i]
        if altura is not None:
            candidatos = [a for a in range(altura - 6, altura + 7) if a % 12 == tonica]
            notas[i] = (min(candidatos, key=lambda a: abs(a - altura)), dur)
            break
    return notas


# ======================================================================
# Saída: MIDI e MP3
# ======================================================================
def salvar_midi(notas, caminho, bpm):
    from music21 import instrument, note, stream, tempo

    parte = stream.Part()
    parte.insert(0, instrument.Violin())
    parte.insert(0, tempo.MetronomeMark(number=bpm))
    for altura, dur in notas:
        parte.append(note.Rest(quarterLength=dur) if altura is None
                     else note.Note(altura, quarterLength=dur))
    stream.Score([parte]).write("midi", fp=str(caminho))


def sintetizar(notas, bpm, timbre, taxa=44100):
    """Sintetizador retrô: onda quadrada (pulso 25%) com envelope curto."""
    seg_por_tempo = 60.0 / bpm
    partes = []
    for altura, dur in notas:
        n = int(round(float(dur) * seg_por_tempo * taxa))
        if altura is None:
            partes.append(np.zeros(n))
            continue
        freq = 440.0 * 2 ** ((altura - 69) / 12)
        t = np.arange(n) / taxa
        if timbre == "triangular":
            onda = 2 * np.abs(2 * ((t * freq) % 1.0) - 1) - 1
        else:
            onda = np.where((t * freq) % 1.0 < 0.25, 1.0, -1.0)
        env = np.ones(n)
        ataque, soltura = min(n, int(0.005 * taxa)), min(n, int(0.03 * taxa))
        env[:ataque] = np.linspace(0, 1, ataque)
        env[n - soltura:] *= np.linspace(1, 0, soltura)  # separa notas repetidas
        env *= np.exp(-1.5 * t)  # decaimento leve
        partes.append(0.3 * onda * env)
    audio = np.concatenate(partes + [np.zeros(int(0.5 * taxa))])
    return (audio * 32767).astype(np.int16), taxa


def salvar_mp3(notas, caminho, bpm, timbre):
    import lameenc

    audio, taxa = sintetizar(notas, bpm, timbre)
    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(taxa)
    enc.set_channels(1)
    enc.set_quality(2)
    Path(caminho).write_bytes(enc.encode(audio.tobytes()) + enc.flush())


# ======================================================================
# Linha de comando
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Gera um reel irlandês com cadeia de Markov.")
    ap.add_argument("--ordem", type=int, default=3, help="ordem n da cadeia (padrão 3)")
    ap.add_argument("--compassos", type=int, default=16,
                    help="número total de compassos (padrão 16; na forma aabb, múltiplo de 4)")
    ap.add_argument("--forma", choices=["livre", "aabb"], default="livre",
                    help="livre: sequência única; aabb: partes A e B repetidas, "
                         "ajustadas a Ré maior e terminando na tônica (padrão livre)")
    ap.add_argument("--timbre", choices=["quadrada", "triangular"], default="quadrada",
                    help="forma de onda do MP3 (padrão quadrada)")
    ap.add_argument("--semente", type=int, default=None,
                    help="semente aleatória; sem ela, uma é sorteada e exibida")
    ap.add_argument("--saida", default="musica", help="nome base dos arquivos .mid e .mp3")
    ap.add_argument("--bpm", type=int, default=180, help="semínimas por minuto (padrão 180)")
    ap.add_argument("--nota-inicial", type=int, default=None,
                    help="altura MIDI da 1ª nota (padrão: a mais comum no corpus)")
    ap.add_argument("--nota-min", type=int, default=None, help="padrão: menor nota do corpus")
    ap.add_argument("--nota-max", type=int, default=None, help="padrão: maior nota do corpus")
    args = ap.parse_args()
    if args.forma == "aabb" and args.compassos % 4 != 0:
        ap.error("na forma aabb, --compassos deve ser múltiplo de 4 (cada parte tem compassos/4)")
 
    musicas = carregar_corpus()
    est = estatisticas_corpus(musicas)
    nota_inicial = args.nota_inicial if args.nota_inicial is not None else est["primeira_comum"]
    nota_min = args.nota_min if args.nota_min is not None else est["min"]
    nota_max = args.nota_max if args.nota_max is not None else est["max"]
    semente = args.semente if args.semente is not None else random.randrange(1_000_000)
 
    modelo = CadeiaMarkov(args.ordem).treinar(para_simbolos(m["notas"]) for m in musicas)
    if args.forma == "aabb":
        notas = gerar_aabb(modelo, semente, nota_inicial, nota_min, nota_max,
                           compassos_parte=args.compassos // 4)
        detalhe = f"partes A e B com {args.compassos // 4} compassos cada"
    else:
        simbolos, ordens = gerar_melodia(modelo, args.compassos, semente,
                                         nota_inicial, nota_min, nota_max)
        notas = para_notas(simbolos, nota_inicial)
        backoff = sum(1 for k in ordens if k < args.ordem) / max(1, len(ordens))
        detalhe = f"{backoff:.0%} dos sorteios usaram backoff"
 
    salvar_midi(notas, f"{args.saida}.mid", args.bpm)
    salvar_mp3(notas, f"{args.saida}.mp3", args.bpm, args.timbre)
 
    print(f"Corpus: {len(musicas)} reels | ordem {args.ordem} | semente {semente} | "
          f"forma {args.forma} | timbre {args.timbre}")
    print(f"Gerados {len(notas)} símbolos em {args.compassos} compassos ({detalhe})")
    print(f"Arquivos: {args.saida}.mid e {args.saida}.mp3")


if __name__ == "__main__":
    main()

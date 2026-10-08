# Gerador de reels irlandeses com cadeias de Markov

Trabalho da disciplina de IA Generativa para Música. O sistema aprende probabilidades de transição a partir de um corpus real de melodias folclóricas irlandesas e gera novas melodias em MIDI e MP3 usando uma cadeia de Markov de ordem n configurável.

## Estrutura

```
markov.py          classe CadeiaMarkov
musica.py          geração do corpus, conversão, geração da música, saída .mid/.mp3 e leitura dos comandos
avaliacao.py       experimentos do trade-off ordem baixa x ordem alta e gráficos
requirements.txt   dependências
exemplos/          melodias geradas com diferentes ordens, compassos, sementes e bpm
graficos/          gráficos e tabela (resultados.csv) da avaliação
```

Na primeira execução é criado o arquivo `corpus_reels.json`, um cache do corpus extraído.

## Instalação

Requer Python 3. Eu rodei na versão 3.12.13.
Para instalar as libs usadas:

```
pip install -r requirements.txt
```

## Uso

### Gerar música

```
python musica.py --ordem 3 --compassos 16 --semente 42 --saida minha_musica
```

Gera `minha_musica.mid` e `minha_musica.mp3`. A primeira execução demora mais por causa da extração do corpus.
As subsequentes são considerávelmente mais rápidas.

| Opção | Padrão | Descrição |
|---|---|---|
| `--ordem` | 3 | ordem n da cadeia (quantos símbolos anteriores são considera) |
| `--compassos` | 16 | compassos da música |
| `--semente` | sorteada | semente aleatória (int). uma será sorteada em caso de valor null |
| `--saida` | `musica` | nome dos arquivos de saída |
| `--bpm` | 180 | batidas por minuto |
| `--nota-inicial` | 74 (Ré5) | altura midi da primeira nota |
| `--nota-min` / `--nota-max` | 55 / 88 | faixa de alturas permitida |

A mesma semente com os mesmos parâmetros gera arquivos idênticos.

### Rodar avaliação

```
python avaliacao.py --ordens 1 2 3 4 5 6 --sementes 30 --compassos 16 --k 8
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--ordens` | 1 a 6 | ordens a ser avaliadas |
| `--sementes` | 30 | quant. musicas geradas por semente |
| `--compassos` | 16 | compassos da música |
| `--k` | 8 | tamanho usado para a taxa de cópia, precisa ser maior que a maior ordem |
| `--pasta` | `graficos` | Pasta de saída dos gráficos e do CSV |

## Funcionamento

### Corpus

Foram usados os 371 reels do livro *O'Neill's Music of Ireland: 1850 Melodies* (números 1176 a 1555), que estão inclusos na lib music21.

### Modelo

A classe `CadeiaMarkov` conta as transições das ordens de 0 a n e calcula as probabilidades por verossimilhança.
Cada sequência do corpus é delimitada pelos símbolos INICIO e FIM.
Backoff: quando um contexto de ordem n não foi visto no treino, ou quando as continuações são restritas, o modelo recorre ao contexto de ordem n-1, n-2, e assim por diante.

### Geração

Durante os sorteio são removidos os candidatos que:
são FIM antes de completar os compassos, ultrapassam o tempo restante ou levam a altura para fora da faixa permitida.

Nesses casos, O sorteio é refeito entre os candidatos restantes.

### Saída

O MIDI usa timbre de violino.
O MP3 é produzido por um sintetizador em Python (onda quadrada com pulso de 25%, de sonoridade retrô) e codificado com lameenc.
# Modelo n-gramas

Este projeto realiza a predição de palavras a partir do **modelo n-gramas** (com suavização de Laplace) a partir de um _corpus_ textual ou utilizando o **modelo de linguagem Tucano**.  
O usuário escolhe via terminal qual abordagem deseja utilizar.

> O projeto foi desenvolvido para disciplina optativa de Processamento de Linguagem Natural.

## Funcionalidades

- Pré-processamento do _corpus_ (.txt)
- Segmentação e tokenização com NLTK
- Geração de unigramas, bigramas e trigramas com suavização de Laplace
- Predição de palavras usando modelo n-gramas
- Geração de palavras com base no modelo **TucanoBR/Tucano-2b4**

## Tecnologias e Bibliotecas

- Python 3
- [NLTK](https://www.nltk.org/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Modelo Tucano](https://huggingface.co/TucanoBR/Tucano-2b4)
- Unidecode

## Corpus
- O _corpus_ utilizado para o desenvolvimento foi a legenda do filme `Ainda Estou Aqui (2024)`.
- É possível utilizar qualquer texto, desde que esteja em `.txt` e renomeado para `corpus_bruto.txt`.
  
## Modelo Tucano
O **Tucano** é um modelo de linguagem voltado ao português brasileiro, treinado pela comunidade. Neste projeto, ele é utilizado para prever a próxima palavra de maneira contextual, servindo como alternativa ao modelo estatístico de n-gramas.

## Desenvolvido por
- [Kailany Alves](https://github.com/kailanyas)
  

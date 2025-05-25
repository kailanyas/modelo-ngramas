import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import defaultdict
from unidecode import unidecode
from transformers import pipeline
import torch

#Carregando o modelo Tucano
generator = pipeline("text-generation", model="TucanoBR/Tucano-2b4")

#Função para leitura de arquivo
def leitura(nome):
    arquivo = open(nome,'r', encoding='utf-8')
    conteudo = arquivo.read()
    arquivo.close()
    return conteudo

def limpar(lista):
    lixo='.,:;?!"\'()[]{}\\/|#$%^&*-'
    quase_limpo = [x.strip(lixo).lower() for x in lista]
    return [x for x in quase_limpo if x.isalpha() or '-' in x]

#Função para o pré-processamento do texto
def pre_processamento(texto):
    #Removendo tags HTML e cabeçalho das legendas (números de fala e tempo)
    texto = re.sub(r'<.*?>|\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', texto)

    #Substituindo quebras de linha por espaço e removendo espaços extras
    texto = re.sub(r'\n+', ' ', texto).strip()

    #Removendo hifens de falas
    texto = re.sub(r'-\s+', '', texto)

    return texto

#Função para segmentação do texto com nltk
def segmentacao(texto):
    sentencas = sent_tokenize(texto)
    return sentencas

#Função para tokenizacao de palavras
def tokenizacao(segmentos):
    tokens = word_tokenize(segmentos)
    return tokens

#Função para o modelo ngramas
def ngramas(n, sent):
	return [tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]

#Calcula a probabilidade de um unigrama
def probabilidade_unigrama(x):
	V = len(vocabulario)
	N = sum(unigramas.values())
	return((unigramas[x] + 1) /  (N + V))

#Calcula a probabilidade de um bigrama
def probabilidade_bigrama(x):
	V = len(vocabulario)
	return((bigramas[x] + 1) / (unigramas[(x[0],)] + V))

#Calcula a probabilidade de um trigrama
def probabilidade_trigrama(x):
     V = len(vocabulario)
     return (trigramas[x] + 1) / (bigramas[(x[0], x[1])] + V)

#Função para predição das 3 palavras mais prováveis (trigramas ou bigramas)
def predicao(palavra1, palavra2=None):
    prever = []
    
    if palavra2:
        lista_tri = [ch for ch in trigramas.keys() if ch[0] == palavra1 and ch[1] == palavra2]

        if lista_tri:
            desc_tri = sorted(lista_tri, key=lambda x: probabilidade_trigrama(x), reverse=True)

            for i in desc_tri[:3]:
                topo = i[2]
                prever.append(topo) 
            return " | ".join(prever)
        else:
            lista_bi = [ch for ch in bigramas.keys() if ch[0] == palavra2]  
            
            if lista_bi:
                desc_bi = sorted(lista_bi, key=lambda x: probabilidade_bigrama(x), reverse=True)
                
                for i in desc_bi[:3]:
                    topo = i[1]
                    prever.append(topo) 
                return " | ".join(prever)
            else:
                return "Não há previsão"
    else:
        lista_bi = [ch for ch in bigramas.keys() if ch[0] == palavra1]

        if lista_bi:
            desc_bi = sorted(lista_bi, key=lambda x: probabilidade_bigrama(x), reverse=True)
                
            for i in desc_bi[:3]:
                topo = i[1]
                prever.append(topo) 
            return " | ".join(prever)
        else:
            return "Não há previsão"
        
#Função de predição com o modelo Tucano
def predicao_tucano(texto):
    completions = generator(texto, num_return_sequences=3, max_new_tokens=1, temperature=1.0)

    palavras = [comp['generated_text'].split()[-1] for comp in completions]
    return " | ".join(palavras)

''' PRÉ-PROCESSAMENTO DO TEXTO PARA UTLIZAR NO MODELO '''
#Abrindo o corpus para leitura e aplicando o pré-processamento
corpus_bruto = leitura("corpus_bruto.txt")
corpus_processado = pre_processamento(corpus_bruto)

#Segmentando o corpus
corpus = segmentacao(corpus_processado)

#Limitando o corpus com tags <s> e </s>
corpus_marcado = [['<s>'] + limpar(s.split()) + ['</s>'] for s in corpus]

#Escrevendo em um arquivo o corpus preparado para treino
arquivo_treino = open('corpus_treino.txt','w')
for i in corpus_marcado:
	arquivo_treino.write(' '.join(i) + '\n')
arquivo_treino.close

''' MODELO DE TRIGRAMAS COM SUAVIZAÇÃO DE LAPLACE '''
#Abrindo o arquivo para utilizar no modelo
arquivo = open('corpus_treino.txt','r')
lista_linhas = arquivo.readlines()
arquivo.close()

#Defindo variaveis para guardar as palavras do corpus e a quantidade em que aparecem
vocabulario = set()
contagens = defaultdict(int)

for l in lista_linhas:
    tokens = tokenizacao(l)   
    for palavra in tokens:
         vocabulario |= {palavra}
         contagens[palavra] += 1

hapax = [p for p in contagens.keys() if contagens[p] == 1]
vocabulario -= set(hapax)
vocabulario |= {'<DES>'}

unigramas = defaultdict(int)
bigramas = defaultdict(int)
trigramas = defaultdict(int)

for l in lista_linhas:
    tokens = l.split() 

    for i in range(len(tokens)):
        if tokens[i] in hapax:
              tokens[i] = '<DES>'
    
    uni = ngramas(1, tokens)
    bi = ngramas(2, tokens)
    tri = ngramas(3, tokens)

    for x in uni:
        unigramas[x] += 1

    for x in bi:
        bigramas[x] += 1
    
    for x in tri:
        trigramas[x] += 1

#print(trigramas)

''' INTERAÇÃO COM USUÁRIO '''
opcao = input("Qual modelo você gostaria de usar?\nDigite um número:\n1 - Modelo n-gramas\n2 - Modelo tucano\n")

if opcao == "1":
    contexto = []

    print("Digite palavra para predição (para finalizar, digite #):")

    while True:
        p = input("").lower().strip()

        if p == "#":
            break  
        else:
            contexto.append(p)
            if len(contexto) >= 2:
                palavra1, palavra2 = contexto[-2], contexto[-1]
                pred = predicao(palavra1, palavra2)
                print(pred)
            elif len(contexto) == 1:
                palavra1 = contexto[-1]
                pred = predicao(palavra1)
                print(pred)
            else:
                print("Não há previsão")
elif opcao == "2":
    print("Digite uma frase para predição (para finalizar, digite #):")

    while True:
        f = input("").lower().strip()

        if f == "#":
            break  
        else:
            pred = predicao_tucano(f)
            print(pred)
else:
    print("Digite uma opção válida!")

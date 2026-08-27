# Indice vetorial e busca

Um banco de dados vetorial armazena vetores e responde consultas do tipo
"quais sao os k vizinhos mais proximos deste vetor". Para colecoes pequenas, ate
alguns milhares de itens, nao e necessario nenhum banco especializado: basta
manter uma matriz numpy em memoria e multiplica-la pelo vetor da pergunta. Essa
busca exaustiva percorre todos os itens e devolve o resultado exato em poucos
milissegundos.

Quando a colecao cresce para milhoes de vetores, a busca exaustiva fica cara e
entram em cena os indices aproximados, conhecidos como ANN. Estruturas como HNSW
e IVF trocam um pouco de precisao por muita velocidade, organizando os vetores em
grafos ou particoes para visitar apenas uma fracao da base.

O limiar de score serve para descartar resultados fracos. Se o melhor trecho
encontrado tem similaridade muito baixa, provavelmente a base nao contem a
resposta, e devolver esse trecho so induz o modelo ao erro. Alem do limiar
absoluto, vale um corte relativo: trechos muito abaixo do melhor resultado sao
enchimento para completar a cota do top-K, nao resposta.

Busca so por vetor erra numa classe inteira de pergunta: a do termo raro e
literal. Um nome proprio que aparece em tres trechos de trezentos se dilui no
vetor, enquanto o BM25, que pesa exatamente o termo que quase ninguem usa, o
encontra de primeira. Fundir os dois sinais cobre o que cada um perde: o denso
acha parafrase, o lexico ancora em numero, sigla e nome proprio.

Trocar o modelo de embedding invalida o indice inteiro. Vetores gerados por
modelos diferentes vivem em espacos distintos e nao podem ser comparados, entao
qualquer mudanca de modelo exige reindexar todos os documentos do zero.

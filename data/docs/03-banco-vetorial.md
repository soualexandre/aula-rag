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
resposta, e devolver esse trecho so induz o modelo ao erro. Nesta aplicacao o
limiar padrao e 0.15 e pode ser ajustado por requisicao.

Trocar o modelo de embedding invalida o indice inteiro. Vetores gerados por
modelos diferentes vivem em espacos distintos e nao podem ser comparados, entao
qualquer mudanca de modelo exige reindexar todos os documentos do zero.

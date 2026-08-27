# Como funciona o RAG

RAG e a sigla de Retrieval Augmented Generation, ou geracao aumentada por
recuperacao. A ideia central e nao depender apenas do que o modelo memorizou
durante o treinamento: antes de responder, o sistema busca trechos relevantes em
uma base propria e injeta esses trechos no prompt.

O pipeline tem duas fases. A fase de indexacao acontece uma vez: os documentos
sao lidos, quebrados em pedacos menores chamados chunks, cada chunk vira um
embedding e todos os vetores sao guardados em um indice. A fase de consulta
acontece a cada pergunta: a pergunta tambem vira um embedding, o indice e
percorrido em busca dos vetores mais proximos, e os melhores chunks formam o
contexto.

O tamanho do chunk e um ajuste delicado. Chunks muito grandes diluem o
significado e desperdicam espaco no prompt. Chunks muito pequenos perdem o
contexto ao redor da informacao. Um ponto de partida comum fica entre 300 e 800
caracteres, com uma sobreposicao de 10% a 20% entre chunks vizinhos para nao
cortar uma ideia no meio.

Recuperar apenas os vetores mais parecidos costuma trazer trechos redundantes,
quase identicos entre si. A tecnica de Maximal Marginal Relevance, ou MMR,
resolve isso penalizando candidatos que se parecem demais com o que ja foi
selecionado, produzindo um contexto mais variado e informativo.

A ultima etapa do RAG e a geracao, em que um LLM le o contexto e escreve a
resposta. Esta aplicacao para de proposito antes dessa etapa: ela entrega o
prompt pronto, com as citacoes numeradas, mas nao chama nenhum modelo de
linguagem.

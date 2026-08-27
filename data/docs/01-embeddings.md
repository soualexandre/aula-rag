# O que sao embeddings

Embeddings sao representacoes numericas de texto. Um modelo de embedding recebe
uma frase e devolve um vetor de numeros reais, tipicamente com 384, 768 ou 1024
dimensoes. Textos com significado parecido produzem vetores proximos no espaco.

A proximidade entre dois vetores e medida com similaridade de cosseno, que
calcula o cosseno do angulo entre eles. O resultado varia de -1 a 1: quanto mais
proximo de 1, mais semanticamente parecidos sao os textos. Quando os vetores sao
normalizados para norma 1, a similaridade de cosseno vira um simples produto
escalar, o que torna a busca extremamente rapida.

Diferente da busca por palavra-chave, a busca por embeddings encontra resultados
mesmo quando nenhuma palavra e igual. A pergunta "como troco minha senha" pode
recuperar um trecho que fala em "redefinicao de credenciais de acesso", porque os
dois textos ocupam regioes vizinhas do espaco vetorial.

O modelo usado nesta demonstracao e o paraphrase-multilingual-MiniLM-L12-v2, que
gera vetores de 384 dimensoes, entende mais de 50 idiomas e ocupa cerca de 220 MB
em disco. Ele roda em CPU via ONNX Runtime, sem precisar de GPU nem de PyTorch.

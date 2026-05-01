# Projeto-IA-Analise-de-Voos
Recentemente desenvolvi um projeto de Rede Neural Artificial para classificar voos como pontuais ou atrasados, e achei interessante compartilhar um pouco de como foi toda a execução do código.

Comecei gerando um conjunto de dados sintético, simulando variáveis importantes como distância do voo, histórico de atrasos, condições climáticas e tráfego aéreo. Essa etapa foi importante para criar um cenário controlado onde eu pudesse testar o modelo.

Depois disso, fiz a separação entre variáveis de entrada (features) e a variável alvo (se o voo está atrasado ou não), além de dividir os dados em treino e teste. Isso garante que o modelo seja avaliado de forma mais justa, usando dados que ele não viu durante o aprendizado.

Na sequência, apliquei a normalização dos dados com o StandardScaler. Esse passo é essencial em redes neurais, pois garante que todas as variáveis estejam na mesma escala, melhorando o desempenho do treinamento.

O modelo utilizado foi um MLPClassifier (Perceptron Multicamadas), que é uma forma de rede neural feedforward. Configurei duas camadas ocultas, permitindo que o modelo capturasse padrões mais complexos nos dados.

Após o treinamento, avaliei o desempenho utilizando a acurácia e, principalmente, a matriz de confusão. A acurácia mostrou um resultado alto, mas a matriz de confusão trouxe uma visão mais completa, evidenciando os acertos e erros do modelo.

O ponto mais interessante foi perceber que o modelo teve um ótimo desempenho na identificação de voos atrasados, evitando erros críticos nesse tipo de problema.

Esse projeto reforçou na prática como cada etapa — desde o tratamento dos dados até a avaliação — é fundamental para construir um modelo de machine learning mais confiável.

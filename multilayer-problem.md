1. Objetivos do Problema

Você deve projetar e implementar Redes PMC para resolver dois problemas de classificação específicos:

    Iris Plants (dataset clássico de flores).

    Problema do Círculo.

2. Configurações Experimentais (Itens 1 e 2)

Você deve realizar dois experimentos principais para encontrar e validar a melhor topologia de rede:
Parâmetros Obrigatórios

    Validação: Uso de 10-fold cross-validation.

    Inicialização: Matrizes de pesos com valores aleatórios entre 0 e 1.

    Função de Ativação: Logística (sigmóide) para todos os neurônios:
    g(u)=1+e−βu1​

    Derivada da Função: g′(u)=β⋅g(u)⋅(1−g(u)).

    Hiperparâmetros: * Taxa de aprendizado (η): 0.5.

        Precisão (ϵ): 10−3.

        Inclinação da sigmóide (β): 0.5.

        Fator de Momentum (α): 0.9 (apenas para o item 2).

Procedimento e Registro

Para cada topologia testada, você deve registrar em uma tabela a média e o desvio padrão de:

    EQM (Erro Quadrático Médio).

    Número de Épocas até a convergência.

    Acurácia (porcentagem de acerto no conjunto de validação).

    Atenção: Para que a comparação entre o Backpropagation padrão e com Momentum seja justa, você deve utilizar exatamente os mesmos dados em cada fold para ambos os casos.

3. Processamento e Teste Final (Item 3)

    Pós-processamento: Implementar uma rotina de arredondamento simétrico para converter as saídas reais da rede em classes inteiras (0 ou 1):

        yi​≥0.5⇒1.

        yi​<0.5⇒0.

    Avaliação Final: Após definir a melhor topologia para cada algoritmo, execute o teste final com os conjuntos de teste e relate a média e o desvio padrão das taxas de acerto.

4. Parte Teórica (Item 4)

Você deve redigir uma explicação detalhada sobre:

    Underfitting e Overfitting: O que são e como detectá-los.

    Soluções: Quais medidas podem ser tomadas para resolver cada uma dessas situações.
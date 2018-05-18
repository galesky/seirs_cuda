# seirs_cuda

## Implementação multi agentes para simulação de propagação de doenças

Esta versão foi abandonada ainda em versões iniciais e migrada para um respositório privado por utilizar mapas proprietários

É possível explorar esta versão rodando o arquivo flattened_list_multilote

O problema abordado é similar ao desenvolvido por KAIZER, 2016 (Uma solução paralela de um modelo multiagente para simulação computacional da propagação de hipotéticas doenças)

O código disponível neste repositório possui as seguintes restrições:
Single Thread - CPU

Há também algumas partes de código em que são abordados shifts circulares de Bits em Inteiros e algumas verificações para envio e manipulação de vetores utilizando CUDA (por consequencia, paralelismo na GPU).

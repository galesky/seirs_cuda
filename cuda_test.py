#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:42:38 2017

@author: eduardo
"""

import numpy as np
import map_reader as mr
import macros
import time
import cuda_jit as cuda
from numba.cuda.random import create_xoroshiro128p_states
import numba

#++++++++++++++IMPORTANDO VIZINHANCAS ++++++
with open('Vizinhancas.csv', 'r') as f:
    linhas = [row.split(';') for row in f]
    
    linhas[0].pop()
    desloc = []
    for i in  linhas[0]:
        desloc.append(i)
    desloc = np.array(desloc, dtype=np.uint16)
        
    linhas[1].pop()
    viz = []
    for i in linhas[1]:
        viz.append(i)
    viz = np.array(viz, dtype=np.uint16)
    
#+++++++++++ IMPORTANDO DADOS DOS info_lotes ++++++++++
info_lotes, total_pos = mr.startLotesGPU()

# cria uma array de 128 ints vazia para cada posição dentro dos info_lotes
pos = np.zeros((total_pos, 128), dtype=np.uint32)

"""                 INICIALIZAÇÃO DO AMBIENTE                   """
n_agentes = macros.qnt_lotes * (macros.ag_por_lote + macros.inf_por_lote)
ag_arr = np.zeros(n_agentes, dtype=np.uint32)

indice = 0
for lote in range(macros.qnt_lotes):
    
    # Informações do lote
    start = info_lotes[3 * lote] # índice inicial do lote na array de posições
    size_x = info_lotes[3 * lote + 1]
    size_y = info_lotes[3 * lote + 2]
    
    # Criação dos agentes
    for i in range(macros.ag_por_lote + macros.inf_por_lote):
        
        # sorteio da posição
        posx = np.random.randint(size_x)
        posy = np.random.randint(size_y)
        
        # adiciona agentes infectados
        status = 0
        if (i >= macros.ag_por_lote):
            status = 2
            
        # construção do bitstring
        ag_arr[indice] = (lote << 26) | (posx << 17) | (posy << 8) | status
        
        # determinação da posição do agente
        pos_agente = start + posx + posy * size_x
        
        n = pos[pos_agente, 0] # total de agentes já naquela posição
        
        if (n < 127): # cabem 127 em cada posição, [0] guarda o total
            n += 1
            pos[pos_agente, 0] = n
            pos[pos_agente, n] = indice + 1
        
        indice += 1
        
"""             TRANSFERINDO DADOS PARA GPU             """

d_desloc = numba.cuda.to_device(desloc)
d_viz = numba.cuda.to_device(viz)
d_ag_arr = numba.cuda.to_device(ag_arr)

# Definição do número de blocos e threads # max de 1024 threads por bloco
blocks = 32
threads = int(np.ceil(n_agentes / blocks))

# Inicialização da array de estados do RNG
nova_seed = np.random.randint(2**30)
rng_states = create_xoroshiro128p_states(blocks * threads, seed=nova_seed)

tempo = []
for ciclo in range(macros.ciclos):
    
    refTime = time.time()
    
    # Chama o kernel CUDA
    cuda.cycle[blocks, threads](d_ag_arr, d_viz, d_desloc, rng_states)
     
    endTime = time.time()
    if (ciclo != 0):
        tempo.append(endTime - refTime)
#    print("Iteração", ciclo, "|", endTime - refTime)
print(np.mean(tempo))
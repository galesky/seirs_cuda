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
    
#+++++++++++ IMPORTANDO DADOS DOS LOTES ++++++++++
lotes, pos_size = mr.startLotesGPU()

# Inicialização dos agentes
ag_arr = []
# cria uma array de 128 ints vazia para cada posição dentro dos lotes
pos = np.zeros((pos_size, 128), dtype=np.uint32)
for lote in range(macros.qnt_lotes):
    start = lotes[3 * lote] # índice inicial do lote na array de posições
    size_x = lotes[3 * lote + 1]
    size_y = lotes[3 * lote + 2]
    for i in range(macros.ag_por_lote + macros.inf_por_lote):
        # sorteio da posição
        posx = np.random.randint(size_x)
        posy = np.random.randint(size_y)
        status = 0
        if (i >= macros.ag_por_lote): # adiciona agentes infectados
            status = 2
        # construção do bitstring
        ag = (lote << 26) | (posx << 17) | (posy << 8) | status
        ag_arr.append(ag)
        #++++ indexação do agente na array de posição ++++#
        index = len(ag_arr) # 0 é vazio, 1 é o primeiro agente
        # determinação da posição do agente
        pos_agente = posx + posy * size_x
        pos_abs = start + pos_agente
        n = pos[pos_abs, 0] # total de agentes já naquela posição
        if (n < 127): # cabem 127 em cada posição, [0] guarda o total
            n += 1
            pos[pos_abs, 0] = n
            pos[pos_abs, n] = index
# transforma a array de agentes em uma np_array para uso no kernel
ag_arr = np.array(ag_arr, dtype=np.uint32)
# calcula o total de blocos e threads a serem usados no kernel
blocks = 32
threads = int(np.ceil(len(ag_arr) / blocks)) # máximo de 1024 threads por bloco
# Inicialização da array de estados do RNG
sd = np.random.randint(2**30) # sorteia uma seed
rng_states = create_xoroshiro128p_states(blocks * threads, seed=sd)
# Transfere as arrays para a GPU
d_desloc = numba.cuda.to_device(desloc)
d_viz = numba.cuda.to_device(viz)
d_ag_arr = numba.cuda.to_device(ag_arr)

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
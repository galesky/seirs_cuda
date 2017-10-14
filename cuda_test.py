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
lotes = mr.startLotesGPU()

# Inicialização dos agentes
ag_arr = []
pos = []
# cria uma array de 128 ints vazia para cada posição dentro dos lotes
ultimo_lote = macros.qnt_lotes - 1
ultimo_start = lotes[3 * ultimo_lote]
ultimo_size_x = lotes[3 * ultimo_lote + 1]
ultimo_size_y = lotes[3 * ultimo_lote + 2]
pos_size = ultimo_start + ultimo_size_x * ultimo_size_y
print("Total de posições:", pos_size)
for i in range(macros.qnt_lotes):
    size = lotes[2 * i] * lotes[2 * i + 1]
    

for lote in range(macros.qnt_lotes):
    size_x = lotes[3 * lote + 1]
    size_y = lotes[3 * lote + 2]
    # cria uma array de 128 ints vazia para cada posição dentro do lote
    pos_lote = []
    for i in range(size_x * size_y):
        pos_lote.append(np.empty(128, dtype=np.uint32))
    pos_lote = np.array(pos_lote)
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
        cel = posx + posy * size_x # posição do agente dentro do lote
        # n = pos[lote][cel][0]
        # pos[lote][cel][n + 1] = len(ag_arr) + 1
        # pos[lote][cel][0] = n + 1
        # TODO adicionar índice do agente na array de posições do lote
ag_arr = np.array(ag_arr, dtype=np.uint32)
n = len(ag_arr)
blocks = 32
threads = int(np.ceil(n / blocks)) # máximo de 1024 threads por bloco
# Inicialização da array de estados do RNG
sd = np.random.randint(2**30) # sorteia uma seed
rng_states = create_xoroshiro128p_states(blocks * threads, seed=sd)
# Transfere as arrays de consulta às vizinhanças para a GPU
d_desloc = numba.cuda.to_device(desloc)
d_viz = numba.cuda.to_device(viz)
for ciclo in range(macros.ciclos):
    
    refTime = time.time()
    
    # Chama o kernel CUDA
    cuda.cycle[blocks, threads](ag_arr, d_viz, d_desloc, rng_states)
     
    endTime = time.time()
    #print("Iteração", ciclo, "|", endTime - refTime)

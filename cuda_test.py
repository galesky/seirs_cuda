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
lotes = mr.startLotes()
print()
for lote in lotes:
    lote.resetPositions()

# Inicialização dos agentes
ag_arr = []
for lote in range(macros.qnt_lotes):
    for i in range(macros.ag_por_lote + macros.inf_por_lote):
        # sorteio da posição
        posx = np.random.randint(lotes[lote].size_x)
        posy = np.random.randint(lotes[lote].size_y)
        status = 0
        if (i >= macros.ag_por_lote): # adiciona agentes infectados
            status = 2
        # construção do bitstring
        ag = (lote << 26) | (posx << 17) | (posy << 8) | status
        ag_arr.append(ag)
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
    print("Iteração de número ", ciclo)
    
    refTime = time.time()
    
    # Chama o kernel CUDA
    cuda.cycle[blocks, threads](ag_arr, d_viz, d_desloc, rng_states)
     
    endTime = time.time()
    print(endTime - refTime)

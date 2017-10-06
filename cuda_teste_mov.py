#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:30:19 2017

@author: eduardo
"""

import cuda_jit as cuda
import numpy as np
import time
from numba.cuda.random import create_xoroshiro128p_states

# Lista as informações do agente
def printAgentInfo(agent):
    lote = (agent >> 26)
    posx = (agent >> 17) & 511
    posy = (agent >> 8) & 511
    cont = (agent >> 2) & 63
    stat = agent & 3
    print(bin(agent), lote, posx, posy, cont, stat)
    
def printInfo(ag_arr):
    for a in range(len(ag_arr)):
        printAgentInfo(ag_arr[a])
    print()

#++++++++++++++IMPORTANDO VIZINHANCAS ++++++
with open('Vizinhancas.csv', 'r') as f:
    linhas = [row.split(';') for row in f]
    
    linhas[0].pop()
    desloc = []
    for i in  linhas[0]:
        desloc.append(i)
    desloc = np.array(desloc, dtype=np.uint32)
        
    linhas[1].pop()
    viz = []
    for i in linhas[1]:
        viz.append(i)
    viz = np.array(viz, dtype=np.uint16)

n = 16
    
# Sorteia as posições, contadores e status
x_arr = np.random.randint(512, size=n)
y_arr = np.random.randint(512, size=n)
cont_arr = np.random.randint(64, size=n)
stat_arr = np.random.randint(4, size=n)

# Constrói os bitstrings com os valores sorteados
ags = []
for i in range(n):
    lote = 2
    posx = x_arr[i]
    posy = y_arr[i]
    cont = cont_arr[i]
    cont = 18
    stat = stat_arr[i]
    stat = 1
    ag = (lote << 26) | (posx << 17) | (posy << 8) | (cont << 2) | (stat)
    ags.append(ag)
ag_arr = np.array(ags, dtype = np.uint32)

# Criação das arrays randômicas de movimento e infecção
inf_arr = np.random.randint(5, size=n, dtype=np.uint16)
float_arr = np.random.uniform(size=n)
float_arr = np.array(float_arr, dtype=np.float32)

blocks = 1
threads = int(np.ceil(n / blocks)) # máximo de 1024 threads por bloco
# Inicialização da array de estados do RNG
sd = np.random.randint(2**30) # sorteia uma seed
rng_states = create_xoroshiro128p_states(blocks * threads, seed=sd)
printInfo(ag_arr)
for i in range(1):
    refTime = time.time()
    # Chama o kernel CUDA
    cuda.cycle[blocks, threads](ag_arr, viz, desloc, rng_states)
    endTime = time.time()
    print(endTime - refTime)
    print()
    printInfo(ag_arr)

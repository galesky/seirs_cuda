#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:22:47 2017

@author: eduardo
"""

from numba import cuda
import numpy as np
from accelerate.cuda import rand

# Shift circular para a direita

@cuda.jit('uint32(uint32, uint32)', device = True)
def ror(ag, n):
    a = ag >> n
    b = ag << (32 - n)
    return a | b
          
# Shift circular para a esquerda

@cuda.jit('uint32(uint32, uint32)', device = True)
def rol(ag, n):
    a = ag << n
    b = ag >> (32 - n)
    return a | b

# Acesso às informações do agente

@cuda.jit('uint32(uint32)', device = True)
def getLote(agent):
    return agent >> 26

@cuda.jit('uint32(uint32)', device = True)
def getPosX(agent):
    return (agent >> 17) & 511

@cuda.jit('uint32(uint32)', device = True)
def getPosY(agent):
    return (agent >> 8) & 511

@cuda.jit('uint32(uint32)', device = True)
def getCont(agent):
    return (agent >> 2) & 63

@cuda.jit('uint32(uint32)', device = True)
def getStat(agent):
    return agent & 3

# Modificação das informações do agente

@cuda.jit('uint32(uint32, uint32)', device = True)
def setLote(agent, lote):
    return rol(((ror(agent, 26) >> 6 << 6) | lote), 26)

@cuda.jit('uint32(uint32, uint32)', device = True)
def setPosX(agent, posx):
    return rol(((ror(agent, 17) >> 9 << 9) | posx), 17)

@cuda.jit('uint32(uint32, uint32)', device = True)
def setPosY(agent, posy):
    return rol(((ror(agent, 8) >> 9 << 9) | posy), 8)

@cuda.jit('uint32(uint32, uint32)', device = True)
def setCont(agent, cont):
    return rol(((ror(agent, 2) >> 6 << 6) | cont), 2)

@cuda.jit('uint32(uint32, uint32)', device = True)
def setStat(agent, stat):
    return (agent >> 2 << 2) | stat

# Retorna o período de mudança de status    

@cuda.jit('int32(int32)', device = True)
def periodoInf(stat):
    # randomizar
    if (stat == 1): # exposição
        return 20
    if (stat == 2): # infectância
        return 35
    if (stat == 3): # recuperação
        return 45
    return 0
             
# Função de movimentação

@cuda.jit('uint32(uint32, uint16)', device = True)
def move(ag, direction):
    posx = getPosX(ag)
    posy = getPosY(ag)
    n = direction # random 0-8
    if (n == 0):
        posx -= 1
        posy += 1
    if (n == 1):
        posy += 1
    if (n == 2):
        posx += 1
        posy += 1
    if (n == 3):
        posx -= 1
    if (n == 4):
        # mudança de lote
        lote = getLote(ag)
        lote += 1
        if (lote > 63):
            lote = 0
        ag = setLote(ag, lote)
    if (n == 5):
        posx += 1
    if (n == 6):
        posx -= 1
        posy -= 1
    if (n == 7):
        posy -= 1
    if (n == 8):
        posx += 1
        posy -= 1
    # colisão com as fronteiras
    if (posx < 0):
        posx = 0
    if (posy < 0):
        posy = 0
    if (posx > 511):
        posx = 511
    if (posy > 511):
        posy = 511
    # salva a nova posição no bitstring
    ag = setPosX(ag, posx)
    ag = setPosY(ag, posy)
    return ag

# TODO Propagação da infecção

# Atualiza info dos agentes

@cuda.jit('uint32(uint32)', device = True)
def update(ag):
    cont = getCont(ag)
    stat = getStat(ag)
    if ((stat != 0) & (cont >= periodoInf(stat))):
        stat = (stat + 1) & 3
        cont = 0
    else:
        if (cont < 63):
            cont += 1
    ag = setStat(ag, stat)
    ag = setCont(ag, cont)
    return ag
    
# Método principal do ciclo

@cuda.jit('void(uint32[:], uint16[:], uint16[:])')
def cycle(ag_arr, mov_arr, inf_arr):
    i = cuda.grid(1)
    ag_arr[i] = move(ag_arr[i], mov_arr[i])
    ag_arr[i] = update(ag_arr[i])
    
# Lista as informações do agente

def info(agent):
    lote = (agent >> 26)
    posx = (agent >> 17) & 511
    posy = (agent >> 8) & 511
    cont = (agent >> 2) & 63
    stat = agent & 3
    print(bin(agent), lote, posx, posy, cont, stat)
    
n = 8
x_arr = np.random.randint(512, size=n)
y_arr = np.random.randint(512, size=n)
cont_arr = np.random.randint(64, size=n)
stat_arr = np.random.randint(4, size=n)

ags = []
mov_arr = np.random.randint(9, size=n, dtype=np.uint16)
inf_arr = np.random.randint(10, size=n, dtype=np.uint16)

for i in range(n):
    lote = 38
    posx = x_arr[i]
    posy = y_arr[i]
    cont = cont_arr[i]
    stat = stat_arr[i]
    ag = (lote << 26) | (posx << 17) | (posy << 8) | (cont << 2) | (stat)
    ags.append(ag)

ag_arr = np.array(ags, dtype = np.uint32)

for a in range(n):
    info(ag_arr[a])
print()
cycle[2, 4](ag_arr, mov_arr, inf_arr)
for a in range(n):
    info(ag_arr[a])
    
print()
print(mov_arr)

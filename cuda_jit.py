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

@cuda.jit('void(uint32[:])')
def move(ag_arr):
    i = cuda.grid(1)
    posx = getPosX(ag_arr[i])
    posy = getPosY(ag_arr[i])
    n = 4 # random 0-8
    if (n == 0):
        posx -= 1
    if (n == 1):
        posx += 1
    if (n == 2):
        posy -= 1
    if (n == 3):
        posy += 1
    if (n == 4):
        posx -= 1
        posy -= 1
    if (n == 5):
        posx -= 1
        posy += 1
    if (n == 6):
        posx += 1
        posy -= 1
    if (n == 7):
        posx += 1
        posy += 1
    if (n == 8):
        # Mudança de lote
        pass
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
    ag_arr[i] = setPosX(ag_arr[i], posx)
    ag_arr[i] = setPosY(ag_arr[i], posy)

# TODO Propagação da infecção

# Atualiza info dos agentes

@cuda.jit('void(uint32[:])')
def update(ag_arr):
    i = cuda.grid(1)
    cont = getCont(ag_arr[i])
    stat = getStat(ag_arr[i])
    if ((stat != 0) & (cont >= periodoInf(stat))):
        stat = (stat + 1) & 3
        cont = 0
    else:
        if (cont < 63):
            cont += 1
    ag_arr[i] = setStat(ag_arr[i], stat)
    ag_arr[i] = setCont(ag_arr[i], cont)
    
# Lista as informações do agente

def info(agent):
    lote = (agent >> 26)
    posx = (agent >> 17) & 511
    posy = (agent >> 8) & 511
    cont = (agent >> 2) & 63
    stat = agent & 3
    print(bin(agent))
    print(lote)
    print(posx)
    print(posy)
    print(cont)
    print(stat)
    
rnd = rand.uniform(size=10,dtype=np.float32,device=True)
for i in range(10):
    print(rnd[i])

lote = 38
posx = 511
posy = 12
cont = 45
stat = 3
ag = (lote << 26) | (posx << 17) | (posy << 8) | (cont << 2) | (stat)

arr = np.array([ag], dtype = np.uint32)
info(arr[0])
move[1, 1](arr)
update[1, 1](arr)
info(arr[0])

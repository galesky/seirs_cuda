#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:22:47 2017

@author: eduardo
"""

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

# TODO Propagação da infecção

# Bitshift circular para a direita
@cuda.jit('uint32(uint32, uint32)', device = True)
def ror(ag, n):
    a = ag >> n
    b = ag << (32 - n)
    return a | b
          
# Bitshift circular para a esquerda
@cuda.jit('uint32(uint32, uint32)', device = True)
def rol(ag, n):
    a = ag << n
    b = ag >> (32 - n)
    return a | b

# Funções de acesso às informações do agente

@cuda.jit(device=True)
def getLote(agent):
    return agent >> 26

@cuda.jit(device=True)
def getPosX(agent):
    return (agent >> 17) & 511

@cuda.jit(device=True)
def getPosY(agent):
    return (agent >> 8) & 511

@cuda.jit(device=True)
def getCont(agent):
    return (agent >> 2) & 63

@cuda.jit(device=True)
def getStat(agent):
    return agent & 3

# Funções de modificação das informações do agente
# Retornam o bitstring com o valor novo

@cuda.jit(device=True)
def setLote(agent, lote):
    return rol(((ror(agent, 26) >> 6 << 6) | lote), 26)

@cuda.jit(device=True)
def setPosX(agent, posx):
    return rol(((ror(agent, 17) >> 9 << 9) | posx), 17)

@cuda.jit(device=True)
def setPosY(agent, posy):
    return rol(((ror(agent, 8) >> 9 << 9) | posy), 8)

@cuda.jit(device=True)
def setCont(agent, cont):
    return rol(((ror(agent, 2) >> 6 << 6) | cont), 2)

@cuda.jit(device=True)
def setStat(agent, stat):
    return (agent >> 2 << 2) | stat

# Retorna o período de mudança aleatório de status    
@cuda.jit(device=True)
def periodoInf(stat, rng_states):
    """
        (base + float aleatorio * peso predefinido)
    """
    i = cuda.grid(1)
    if (stat == 1): # exposição
        return 15 + int(5 * xoroshiro128p_uniform_float32(rng_states, i))
    if (stat == 2): # infectância
        return 30 + int(5 * xoroshiro128p_uniform_float32(rng_states, i))
    if (stat == 3): # recuperação
        return 40 + int(5 * xoroshiro128p_uniform_float32(rng_states, i))
    return 0

"""
    TROCA DE LOTE
    
    Procura elementos compatíveis na array de adjacências do lote
    A array apresenta os dados da forma:
        viz[n]      = x_inicial
        viz[n + 1]  = y_inicial
        viz[n + 2]  = x_destino
        viz[n + 3]  = y_destino
        viz[n + 4]  = lote_destino
    Sendo assim deve-se comparar x e y do agente com viz[n] e viz[n + 1]
    A variável qtde armazena o total de elementos compatíveis
    Multiplica-se a qtde por um float aleatório para decidir o destino

"""

@cuda.jit(device=True)
def changeLote(ag, viz, va, vb, rng_states):
    x = getPosX(ag)
    y = getPosY(ag)
    qtde = 0
    v = va
    while (v < vb):
        if (x == viz[v] and y == viz[v + 1]):
            qtde += 1
        v += 5
    if (qtde == 0):
        return ag
    # sorteio do local de destino
    i = cuda.grid(1)
    destino = int(qtde * xoroshiro128p_uniform_float32(rng_states, i))
    j = 0
    v = va
    while (v < vb):
        if (x == viz[v] and y == viz[v + 1]):
            if (j == destino):
                break
            j += 1
        v += 5
    ag = setPosX(ag, viz[v + 2])
    ag = setPosY(ag, viz[v + 3])
    ag = setLote(ag, viz[v + 4])
    return ag
          
"""
    MOVIMENTAÇÃO
    Cada valor posssível corresponde a um espaço na vizinhança de Moore (r=1)
    
        0 1 2
        3   4
        5 6 7           8 -> Kernel chama a função de mudança de lote
        
"""
@cuda.jit(device=True)
def move(ag, direction):
    posx = getPosX(ag)
    posy = getPosY(ag)
    n = direction # random 0-8
    if (n == 0):
        posx -= 1
        posy -= 1
    if (n == 1):
        posy -= 1
    if (n == 2):
        posx += 1
        posy -= 1
    if (n == 3):
        posx -= 1
    if (n == 4):
        posx += 1
    if (n == 5):
        posx -= 1
        posy += 1
    if (n == 6):
        posy += 1
    if (n == 7):
        posx += 1
        posy += 1
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

"""
    UPDATE DO STATUS
    
"""

# Atualiza info dos agentes
@cuda.jit(device=True)
def update(ag, rng_states):
    cont = getCont(ag)
    stat = getStat(ag)
    if ((stat != 0) & (cont >= periodoInf(stat, rng_states))):
        stat = (stat + 1) & 3
        cont = 0
    else:
        if (cont < 63):
            cont += 1
    ag = setStat(ag, stat)
    ag = setCont(ag, cont)
    return ag

# Método principal do ciclo
@cuda.jit
def cycle(ag_arr, viz, desloc, rng_states):
    i = cuda.grid(1)
    direcao_mov = int(9 * xoroshiro128p_uniform_float32(rng_states, i))
    if (direcao_mov == 8):
        # mudança de lote
        lote = getLote(ag_arr[i])
        if (lote < len(desloc)):
            va = desloc[lote]
            vb = desloc[lote + 1]
            ag_arr[i] = changeLote(ag_arr[i], viz, va, vb, rng_states)
    else:
        # movimento aleatório
        ag_arr[i] = move(ag_arr[i], direcao_mov)
    # tick no contador de ciclos individual
    ag_arr[i] = update(ag_arr[i], rng_states)

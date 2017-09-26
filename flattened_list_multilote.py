#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:04:54 2017

@author: galesky
"""

import numpy as np
import time as time
import sys
import csv
import matplotlib.pyplot as plt
import map_reader as map
import macros as macros
import scatter_plot as sc

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"


ag_arr = [] #vetor agentes

vizinhancas = []

deslocamentos_vizinhancas = []


def taxaInfeccao():
    return (np.random.uniform(90,95))
def periodoExposicao():
    return (np.random.uniform(15,20))
def periodoInfectancia():
    return (np.random.uniform(30,35))
def periodoRecuperacao():
    return (np.random.uniform(40,45))

#flattened list
def get_element(elements, x, y, size_x):
    # Get element with two coordinates.
    return elements[x + (y * size_x)]

def set_element(elements, x, y, size_x, value):
    # Set element with two coordinates.
    elements[x + (y * size_x)].append(value)

class agente:
    def __init__ (self, pos_x, pos_y, lote):
        self.x = pos_x #posx 
        self.y = pos_y #posy 
        self.c = 0 #cicles
        self.state = 0 #estado
        self.lote = lote
    
    def addCicle(self):
        self.c += 1
    
    def zeroCicle(self):
        self.c = 0
    
    def changeState(self, new_state):
        self.state = new_state #state
        
    def move(self,mov_x,mov_y):
        self.x = self.x + mov_x
        self.y = self.y + mov_y
    
    def updateLote(self, lote, mov_x, mov_y):
        self.lote = lote
        self.x = mov_x
        self.y = mov_y

"""
Considerou-se que a quadra possui 9 lotes (3x3) e portanto
as vizinhanças possuem 8 possibilidades.

O vetor de vizinhanças representa as posições dos vizinhos da esquerda para
direita e de cima para baixo, no centro(4) por exemplo seria:
    [0,1,2,3,5,6,7,8]

No vetor de vizinhança o id 10 significa que NÃO HÁ vizinhos naquela pos.
"""
#++++++++++++++IMPORTANDO VIZINHANCAS ++++++
with open('Vizinhancas.csv', 'r') as f:
    linhas = [row.split(';') for row in f]
    linhas[0].pop()
    deslocamentos_vizinhancas = linhas[0]
    linhas[1].pop()
    vizinhancas = linhas[1]


#+++++++++++ IMPORTANDO DADOS DOS LOTES ++++++++++
quadra = map.startLotes()
map.printer(quadra)
for lote in quadra:
    print("x pos " + str(lote.size_x) + " y pos " + str(lote.size_y))
    lote.resetPositions()

#novo start - utiliza dados do csv

def startAgents():
    for id_lote in range(macros.qnt_lotes):
        for j in range(macros.ag_por_lote):
            ag_arr.append(agente(np.random.randint(0,quadra[id_lote].size_x),np.random.randint(0,quadra[id_lote].size_y),id_lote))
            id_arr = j + (id_lote * (macros.ag_por_lote + macros.inf_por_lote))
            local_xy = [ag_arr[id_arr].x, ag_arr[id_arr].y]
            set_element(quadra[id_lote].positions, local_xy[0], local_xy[1],quadra[id_lote].size_x, id_arr)
        for firstAgente in range(macros.inf_por_lote): #GERA OS PRIMEIROS AGENTES INFECTADOS
            ag_arr.append(agente(np.random.randint(0, quadra[id_lote].size_x), np.random.randint(0, quadra[id_lote].size_y),id_lote))
            ag_arr[id_lote*(macros.ag_por_lote + macros.inf_por_lote) + firstAgente].changeState(2)


#+++++++++++++INPUTS INICIAIS - TERRENO 3X3
"""def startAgents(id_lote):    
    for i in range(size):
        ag_arr.append(agente(np.random.randint(0,quadra[id_lote].size_x),np.random.randint(0,quadra[id_lote].size_y),id_lote))
        local_xy = [ag_arr[i].x,ag_arr[i].y]
        j = i + (id_lote * size)
        set_element(quadra[id_lote].positions, local_xy[0], local_xy[1], j)
        if np.random.randint(0,100) < 10:
            ag_arr[j].changeState(2)
            
"""
#+
#+++++++++++ITERAÇÕES
def checkPos(id_lote):
    for i in quadra[id_lote].positions: #Uma coordenada do lote
        if len(i) > 1:
            for k in i:
                if ag_arr[k].state == 2: #Se a posição possui um infectado
                    checkPosState(i)     #Passa o vetor de agentes da posição para a função, ex (0,0)
                    break

def checkPosState(pos_arr):
    for i in pos_arr: #varre uma posicao da matriz (vetor de agentes)
        if ag_arr[i].state == 0 and np.random.uniform(0,100) < taxaInfeccao():
            ag_arr[i].changeState(1)

def move():
    cont = 0 # id do lote
    for i in range(macros.qnt_lotes):
        quadra[i].resetPositions()
    for i in ag_arr:
        direcao = np.random.randint(0,9)
        if direcao == 0:
            px = -1
            py = -1
        if direcao == 1:
            px = 0
            py = -1
        if direcao == 2:
            px = 1
            py = -1
        if direcao == 3:
            px = -1
            py = 0
        if direcao == 4:
            px = 1
            py = 0
        if direcao == 5:
            px = -1
            py = 1
        if direcao == 6:
            px = 0
            py = 1
        if direcao == 7:
            py = 1
            px = 1
        if direcao == 8:
            changeLote(i)
        #VALIDAÇÃO SE NÃO ESTÁ FORA DA MATRIZ
        if direcao != 8:
            if ((i.x + px) >= 0) and ((i.x + px) < quadra[i.lote].size_x) and ((i.y + py) >= 0) and ((i.y + py) < quadra[i.lote].size_y):
                i.move(px,py)
        set_element(quadra[i.lote].positions,i.x,i.y,quadra[i.lote].size_x,cont) #alimenta o vetor de contatos, cont é o id do lote

        cont += 1

# noinspection PyTypeChecker
def changeLote(agente):
    quantidade = 0
    posicoes = []
    v_start = int(deslocamentos_vizinhancas[agente.lote]) #posicao de inicio do lote no vetor vizinhancas
    v_end =  int(deslocamentos_vizinhancas[agente.lote + 1]) #posicao de fim do lote no vetor vizinhancas
    while v_start < v_end: #enquanto nao tiver varrido todas as posicoes do lote
        if agente.x == int(vizinhancas[v_start + 0] and agente.y == int(vizinhancas[v_start + 1])):
            quantidade += 1
        v_start += 5
    v_start = int(deslocamentos_vizinhancas[agente.lote]) #reseta a posicao inicial do lote
    if quantidade > 0: #se encontrou algum resultado
        while v_start < v_end:
            if agente.x == int(vizinhancas[v_start + 0] and agente.y == int(vizinhancas[v_start + 1])):
                posicoes.append(int(vizinhancas[v_start + 2])) #x destino
                posicoes.append(int(vizinhancas[v_start + 3])) #y destino
                posicoes.append(int(vizinhancas[v_start + 4])) #lote destino
            v_start += 5
        n_picker = np.random.randint(0,quantidade) #variavel que irá selecionar a posicao de transicao do agente
        agente.updateLote(posicoes[n_picker*3+2],posicoes[n_picker*3],posicoes[n_picker*3+1])
        macros.num_trocas += 1

def updateState():
    hits = 0
    for i in ag_arr: #Varre todos os agentes
        if i.state == 1: #EXPOSTO -> INFECTADO
            if i.c >= periodoExposicao():
                i.changeState(2)
                i.zeroCicle()
            else:
                i.addCicle()
            continue
        if i.state == 2: #INFECTADO -> RECUPERADO
            if i.c >= periodoInfectancia():
                i.changeState(3)
                i.zeroCicle()
            else:
                i.addCicle()
            continue
        if i.state == 3:# RECUPERADO -> SUSCETÍVEL
            if i.c >= periodoRecuperacao():
                i.changeState(0)
                i.zeroCicle()
                hits += 1
            else:
                i.addCicle()
            continue
        hits += 1
    print("SUCETIVES -> ", hits)
#OUTPUTS
def countStatus():
    qnt_status = [0,0,0,0]
    for i in ag_arr:
        qnt_status[i.state] += 1
    return qnt_status

def countLote():
    qnt_lote = []
    for i in range(macros.qnt_lotes):
        qnt_lote.append(0)
    for i in ag_arr:
        qnt_lote[i.lote] += 1
    return qnt_lote

def total(data_arr):
    result = 0
    for i in data_arr:
        result += i
    return result
#FIM - OUTPUTS

""" Iniciando o processo de infecção

1. Inicia os agentes e estabelece o ambiente (lotes)
2. Inicia a lib para gerar o csv com os outputsun 
3. Roda os ciclos
    3.1. Atualiza os contatos entre agentes
    3.2. Movimento os agentes
    3.3. Roda o ciclo da doença (S-> E -> I -> R -> S)
4. Fecha e salva o csv com os outputs

"""

startAgents() #seta o ambiente

for m in range(macros.num_testes):
    outputFile = open('output'+ str(m)+'.csv', 'w')
    outputWriter = csv.writer(outputFile, delimiter=';')
    for n in range(macros.ciclos):
        print ("Iteração de número",n)
        sys.stdout.write(GREEN)
        seirs_data = countStatus()
        lote_data = countLote()
        print ("[S,E,I,R]", seirs_data, "total de agentes:", total(seirs_data))
        print ("total por lote", lote_data, "total nos lotes:", total(lote_data))

        sys.stdout.write(CYAN)
        start = time.time()
        outputWriter.writerow([n] + countStatus()) #rodando em tempo de execução, vetorizar
        move() #3.2

        for i in range(macros.qnt_lotes): #3.1
            checkPos(i)

        updateState()  # 3.3




        print ("tempo de processamento = ", (time.time() - start))
        sys.stdout.write(RESET)
    outputFile.close()  # 4

print ("FIM DA EXECUÇÃO")
print ("NUMERO DE TROCAS DE LOTE : " + str(macros.num_trocas))
sys.stdout.write(RED)

#print("MOVIMENTOS [cima,esq,dir,baixo]")
#print (mov_dir)


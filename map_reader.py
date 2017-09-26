#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:38:06 2017

@author: galesky
"""
import numpy as np
import csv
import os
import macros

quadra = []

class lote:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.positions = []
    def resetPositions(self):
        self.positions = []
        for i in range(self.size_x*self.size_y):
            self.positions.append([]) #criando um array vazio com as posicoes do lote

def startLotes():
    for i in range(0, macros.qnt_lotes):
        caminho = "Lote_" + str(i) + ".csv"
        print(caminho)
        with open(os.path.join('Entradas', 'MonteCarlo_0', caminho), 'r') as f:
            linhas = [row.split(';') for row in f]
            num_linhas = (int(round(float(linhas[1][0]),0)))
            num_colunas = (int(round(float(linhas[2][0]),0)))
            quadra.append(lote(num_linhas,num_colunas))
    return quadra


def printer(int_quadra):
    for i in range(macros.qnt_lotes):
        print (str(i) + " " + str(int_quadra[i].size_x))
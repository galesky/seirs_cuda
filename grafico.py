#!/usr/bin/python3

from sys import argv
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt

def criar_grafico(nome_arquivo):
    with open(nome_arquivo, 'r') as arquivo:
        linhas = [linha.split(';') for linha in arquivo]
        x = [int(i[0]) for i in linhas]
        s = [int(i[1]) for i in linhas]
        e = [int(i[2]) for i in linhas]
        i = [int(i[3]) for i in linhas]
        r = [int(i[4]) for i in linhas]
        plt.plot(x, s, color="green", label="S", linewidth=2)
        plt.plot(x, e, color="yellow", label="E", linewidth=2)
        plt.plot(x, i, color="red", label="I", linewidth=2)
        plt.plot(x, r, color="blue", label="R", linewidth=2)
        plt.legend(loc="upper right")

def plot(arquivo):
    criar_grafico(str(arquivo.absolute()))
    plt.savefig(str(arquivo.absolute()).replace(".csv", ".png"), dpi=400)
    plt.close()

if len(argv) == 2:
    nome_arquivo = argv[1]
    if Path(nome_arquivo).is_dir():
        pool = Pool()
        pool.map(plot, Path(nome_arquivo).rglob('*.csv'))
    else:
        criar_grafico(nome_arquivo)
        plt.show()
        plt.close()
        
else:
    print("Use:")
    print("python Gerador_Saidas_Quantidades.py diretorio\t\t\t")
    print("Para cada arquivo csv dentro do diretorio gera uma imagem png")
    print("python Gerador_Saidas_Quantidades.py arquivo.csv\t\t")
    print("Exibe o grafico de um arquivo csv em tela")

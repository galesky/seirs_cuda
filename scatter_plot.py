
import numpy as np
import time as time
import sys
import csv
import matplotlib.pyplot as plt
import map_reader as map
import macros as macros

def createScatterPlot(ag_arr,id_lote,ciclo):
    fig, ax = plt.subplots()
    for i in ag_arr:  # PLOT DO GRÁFICO , MIRRORED
        if i.lote == id_lote:
            if i.state == 2:
                color = "red"
            elif i.state == 1:
                color = "yellow"
            elif i.state == 0:
                color = "green"
            else:
                color = "blue"
            scale = 50.0

            ax.scatter(i.x , i.y, c=color, s=scale,
                       alpha=0.3, edgecolors='black')
            ax.legend()

    ax.grid(True)
    #plt.show()
    fig.savefig("lote."+str(id_lote)+ str(ciclo) + ".png")
    plt.close(fig)




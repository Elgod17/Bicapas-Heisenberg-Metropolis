'ESTE CODIGO SIMULA UN SISTEMA DE ESPINES HEISENBERG EN UN RED CUADRADA XY'

import numpy as np
import random  
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from numba import njit
import concurrent.futures
executor = concurrent.futures.ProcessPoolExecutor()
executor.shutdown(wait=True, cancel_futures=True)
import gc
gc.collect()




@njit
def random_spin():
    'ALGORITMO DE MORSAGLIA'
    while True:
        r1 = random.uniform(-1, 1)
        r2 = random.uniform(-1, 1)
        rsumsquare = r1**2 + r2**2
        if rsumsquare < 1:
            zeta1 = 1 - 2*r1
            zeta2 = 1 - 2*r2
            Sx = 2 * zeta1 * np.sqrt(1 - rsumsquare)
            Sy = 2 * zeta2 * np.sqrt(1 - rsumsquare)
            Sz = 1 - 2 * rsumsquare

            return np.array([Sx, Sy, Sz])




def config_ordenada(l,k):
    'CONFIGURACION CON TODOS LOS ESPINES ARRIBA'
    config = np.zeros((l+2,k+2,3))
    config[1:l+1, 1:k+1, :] = np.array([0.0, 1.0, 0.0])
    return config



@njit
def metropolis(config_1,T,pasos_locales,l,k):
  'EVOLUCIONAR ESTADO DE LA CONFIGURACION EN pasos_locales de monte carlo'
  config = config_1
  n = np.arange(1,l+1,1)
  m = np.arange(1,k+1,1)

  for _ in range(pasos_locales):
    S = random_spin()
    i = np.random.choice(n)
    j = np.random.choice(m)
    vecinos = config[i+1,j] + config[i-1,j] + config[i,j+1] + config[i,j-1]
    dE = np.dot(config[i,j]-S,vecinos)
    if dE < 0 or np.random.uniform(0,1) < np.exp(-dE/T):
      config[i,j] = S

  Mvec = np.sum(config, axis = 2) ** 2
  M = np.sqrt(np.sum(Mvec))
  return M



def m_vs_t(Ts, configuracion, pasos_termalizacion, pasos,l,k):
  'HACER BARRIDOS DE TEMPERATURA'
  print(configuracion)
  # termalizacion
  Ms = []
  M = metropolis(configuracion,Ts[0],pasos_termalizacion,l,k)
  Ms.append(M)
  for T in Ts[1:]:
    M = metropolis(configuracion,T,pasos,l,k)
    Ms.append(M)
  return Ms


Ts = np.arange(0.0001, 20, 0.2)
MM = m_vs_t(Ts, config_ordenada(10,10), 100000, 20000,20,20)
# Graficar
plt.figure(figsize=(8, 5))
plt.grid()
plt.plot(Ts, MM, 'o-', markersize=4, label='Bicapa Heisenberg')
plt.show()
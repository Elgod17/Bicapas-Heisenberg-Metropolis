'ESTA ES UNA OPTIMICIZACIÓN DEL CÓDIGO "xy_heisenberg.py" HECHA Y COMENTADA POR GEMINI 3'

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


# 1. Generador de spin corregido (Marsaglia)
@njit
def random_spin():
    while True:
        # Generar en rango [-1, 1] directamente
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        s = u**2 + v**2
        
        if s < 1:
            root = np.sqrt(1 - s)
            x = 2 * u * root
            y = 2 * v * root
            z = 1 - 2 * s
            return np.array([x, y, z])




# 2. Configuración ordenada con PADDING (l+2, k+2)
def config_ordenada(l, k):
    # Creamos array con bordes extra para condiciones de frontera
    config = np.zeros((l+2, k+2, 3))
    # Llenamos solo el interior (del 1 al l, del 1 al k)
    # Apuntando todos hacia arriba en Y (vector 0, 1, 0)
    config[1:l+1, 1:k+1, :] = np.array([0.0, 1.0, 0.0])
    return config

# 3. Metrópolis optimizado con Numba
@njit
def metropolis(config, T, pasos_locales, l, k):
    # No necesitamos copiar config si vamos a modificarlo in-place y devolver la magnetización
    # Pero si quieres no dañar el original fuera de la función, hacemos copy:
    cfg = config.copy() 
    
    # Pre-calcular indices válidos (interior de la red)
    # Range en python es [start, stop), así que l+1 es el límite
    rango_i = np.arange(1, l+1)
    rango_j = np.arange(1, k+1)

    for _ in range(pasos_locales):
        # Elegir sitio al azar
        i = np.random.choice(rango_i)
        j = np.random.choice(rango_j)
        
        S_new = random_spin()
        S_old = cfg[i, j]
        
        # Suma vectorial de los 4 vecinos
        vecinos = cfg[i+1, j] + cfg[i-1, j] + cfg[i, j+1] + cfg[i, j-1]
        
        # Cambio de energía: -(S_new - S_old) * Vecinos
        # dE = E_new - E_old
        # E = -S * Vecinos
        dE = -np.dot((S_new - S_old), vecinos)
        
        # Criterio de Metropolis
        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[i, j] = S_new

    # Calcular Magnetización Total del sistema (Promedio de la norma del vector suma)
    # OJO: Solo sumamos el interior, no los bordes (que son cero)
    
    # Sumamos todos los vectores para obtener el Vector Magnetización Total
    M_vec = np.zeros(3)
    for x in range(1, l+1):
        for y in range(1, k+1):
            M_vec += cfg[x, y]
            
    # Retornamos la magnitud de la magnetización promedio por espín
    M_norm = np.sqrt(M_vec[0]**2 + M_vec[1]**2 + M_vec[2]**2)
    return M_norm / (l*k), cfg

# 4. Loop de Temperaturas
def m_vs_t(Ts, config_inicial, pasos_termalizacion, pasos_medicion, l, k):
    Ms = []
    
    # Usamos la configuración actual para la siguiente temperatura (recocido)
    curr_config = config_inicial
    
    # 1. Termalización inicial (calentamiento) a la primera T
    print(f"Termalizando a T={Ts[0]}...")
    mag, curr_config = metropolis(curr_config, Ts[0], pasos_termalizacion, l, k)
    Ms.append(mag)
    
    # 2. Barrido de temperaturas
    for i, T in enumerate(Ts[1:]):
        # Usamos la configuración resultante anterior como inicio de esta
        mag, curr_config = metropolis(curr_config, T, pasos_medicion, l, k)
        Ms.append(mag)
        
        # Barra de progreso simple
        if i % 10 == 0:
            print(f"T = {T:.2f} -> M = {mag:.4f}")
            
    return Ms

# --- EJECUCIÓN ---

# Parámetros
L, K = 60, 60
Pasos_Term = 200000  # Más pasos para termalizar bien
Pasos_Med = 20000    # Pasos por temperatura
Temperaturas = np.arange(0.1, 5.0, 0.1) # Rango típico para ver transición de fase

# Configuración inicial
config_ini = config_ordenada(L, K)

# Correr simulación
magnetizaciones = m_vs_t(Temperaturas, config_ini, Pasos_Term, Pasos_Med, L, K)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(Temperaturas, magnetizaciones, 'o-', markersize=4, label='Bicapa Heisenberg')
plt.title(f"Magnetización vs Temperatura (Red {L}x{K})")
plt.xlabel("Temperatura (T)")
plt.ylabel("Magnetización Promedio |M|")
plt.grid(True)
plt.legend()
plt.show()
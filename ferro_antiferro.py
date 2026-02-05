'ESTE CODIGO SIMULA LA BICAPA FERROMAGNETICO/ANTIFERROMAGNETICO'
'GEMINI 3 HIZO LA ADAPTACION COMPLETA DEL CODIGO "bicapa_ferro_paramagnetico.py" PARA'
'ESTA VERSION EN LA QUE LA SEGUNDA CAPA ES ANTIFERROMAGNETICA'

import numpy as np
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numba import njit, config as numba_config
import concurrent.futures
import multiprocessing
import gc
import time

# Configuración para evitar conflictos de hilos
numba_config.THREADING_LAYER = 'workqueue'

# -----------------------------------------------------------------------------
# 1. FUNCIONES PRINCIPALES (NUMBA)
# -----------------------------------------------------------------------------

@njit
def random_spin():
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        s = u**2 + v**2
        if s < 1:
            root = np.sqrt(1 - s)
            x = 2 * u * root
            y = 2 * v * root
            z = 1 - 2 * s
            return np.array([x, y, z])

@njit
def config_ordenada(l, k, h):
    config = np.zeros((l+2, k+2, h+2, 3))
    # Interior ferromagnético
    config[1:l+1, 1:k+1, 1:h+1, :] = np.array([0.0, 0.0, 1.0])
    return config

@njit
def calcular_energia_total(cfg, l, k, h, J, J_interaccion):
    E = 0.0
    for x in range(1, l + 1):
        for y in range(1, k + 1):
            for z in range(1, h + 1):
                S = cfg[x, y, z]
                if x < l: E -= J * np.dot(S, cfg[x+1, y, z])
                if y < k: E -= J * np.dot(S, cfg[x, y+1, z])
                if z < h: E -= J_interaccion * np.dot(S, cfg[x, y, z+1])
                if z == h: E += J * np.dot(S, cfg[x, y, z+1])
    return E

@njit
def metropolis(config, T, pasos_locales, l, k, h, J, J_interaccion):
    cfg = config
    
    # --- TERMALIZACIÓN ---
    limit = int(pasos_locales / 2)
    for _ in range(limit):
        x = np.random.randint(1, l + 1)
        y = np.random.randint(1, k + 1)
        z = np.random.randint(1, h + 1)
        
        S_new = random_spin()
        S_old = cfg[x,y,z]
        
        # Vecinos y campo
        vec_sum = np.zeros(3)
        vec_sum += cfg[x+1, y, z] + cfg[x-1, y, z] + cfg[x, y+1, z] + cfg[x, y-1, z]
        campo = J * vec_sum
        
        vec_z = np.zeros(3)
        if z > 1: vec_z += cfg[x, y, z-1]
        if z < h: vec_z += cfg[x, y, z+1]
        campo += J_interaccion * vec_z
        
        dE = -np.dot((S_new - S_old), campo)

        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[x,y,z] = S_new

    # --- MEDICIÓN ---
    # Calculamos energía inicial para ir actualizándola (más rápido que recalcular todo)
    E_actual = calcular_energia_total(cfg, l, k, h, J, J_interaccion)
    E_sum = 0.0
    E2_sum = 0.0
    n_medidas = 0
    
    for _ in range(limit):
        x = np.random.randint(1, l + 1)
        y = np.random.randint(1, k + 1)
        z = np.random.randint(1, h + 1)
        
        S_new = random_spin()
        S_old = cfg[x,y,z]
        
        vec_sum = np.zeros(3)
        vec_sum += cfg[x+1, y, z] + cfg[x-1, y, z] + cfg[x, y+1, z] + cfg[x, y-1, z]
        campo = J * vec_sum
        
        vec_z = np.zeros(3)
        if z > 1: vec_z += cfg[x, y, z-1]
        if z < h: vec_z += cfg[x, y, z+1]
        campo += J_interaccion * vec_z
        
        dE = -np.dot((S_new - S_old), campo)
        
        accepted = False
        if dE < 0:
            accepted = True
        elif np.random.random() < np.exp(-dE/T):
            accepted = True
            
        if accepted:
            cfg[x,y,z] = S_new
            E_actual += dE
            
        # Acumular datos para el calor específico
        E_sum += E_actual
        E2_sum += E_actual**2
        n_medidas += 1

    # --- CÁLCULOS FINALES (ESCALARES) ---
    
    # Magnetización Paramagnética (z=1)
    mz_para_sum = 0.0
    for i in range(1, l+1):
        for j in range(1, k+1):
            mz_para_sum += cfg[i, j, 1, 2] # Componente Z
    Mz_para = mz_para_sum / (l * k)

    # Magnetización Ferromagnética (z>1)
    mz_ferro_sum = 0.0
    vol_ferro = l * k * (h - 1)
    if h > 1:
        for z_idx in range(2, h+1):
            for i in range(1, l+1):
                for j in range(1, k+1):
                    mz_ferro_sum += cfg[i, j, z_idx, 2]
        Mz_ferro = mz_ferro_sum / vol_ferro
    else:
        Mz_ferro = 0.0

    # Calor Específico c = (<E^2> - <E>^2) / (N * T^2)
    E_avg = E_sum / n_medidas
    E2_avg = E2_sum / n_medidas
    var_E = E2_avg - (E_avg**2)
    
    N_total = l * k * h
    c = var_E / (N_total * (T**2))

    return Mz_para, Mz_ferro, c

# -----------------------------------------------------------------------------
# 2. GESTIÓN DE SIMULACIÓN
# -----------------------------------------------------------------------------

def m_vs_t(Ts, pasos_termalizacion, pasos_medicion, l, k, h, J, J_interaccion):
    Ms_p = []
    Ms_f = []
    cs = []
    curr_config = config_ordenada(l,k,h)
    
    for i, T in enumerate(Ts):
        pasos = pasos_termalizacion if i == 0 else pasos_medicion
        mz_p, mz_f, c = metropolis(curr_config, T, pasos, l, k, h, J, J_interaccion)
        Ms_p.append(mz_p)
        Ms_f.append(mz_f)
        cs.append(c)
        
    return np.array(Ms_p), np.array(Ms_f), np.array(cs)



def simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, l, k, h, J, J_interaccion):
    Ts = np.arange(Tmin, Tmax + deltaT, deltaT)
    if Ts[0] < 1e-4: Ts = Ts[1:]
    
    print(f"-> Simulando J_int = {J_interaccion}...")
    
    msp, msf, allcs = [], [], []
    
    # Contexto Spawn para evitar cuelgues
    ctx = multiprocessing.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(m_vs_t, Ts, pasos_termalizacion, pasos_medicion, l, k, h, J, J_interaccion) for _ in range(num_replicas)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                mp, mf, c = future.result()
                msp.append(mp)
                msf.append(mf)
                allcs.append(c)
                completed += 1
            except Exception as e:
                print(f"Error: {e}")

    # Convertir a numpy arrays
    mat_msp = np.array(msp)
    mat_msf = np.array(msf)
    mat_cs = np.array(allcs)
    
    # Promedios (Usamos ABS para magnetización para ver orden)
    msp_mean = np.mean(np.abs(mat_msp), axis=0)
    msf_mean = np.mean(np.abs(mat_msf), axis=0)
    cs_mean = np.mean(mat_cs, axis=0) # Calor específico directo
    
    msp_err = np.std(np.abs(mat_msp), axis=0) / np.sqrt(num_replicas)
    msf_err = np.std(np.abs(mat_msf), axis=0) / np.sqrt(num_replicas)
    cs_err = np.std(mat_cs, axis=0) / np.sqrt(num_replicas)
    
    return Ts, msp_mean, msp_err, msf_mean, msf_err, cs_mean, cs_err


# -----------------------------------------------------------------------------
# 3. EJECUCIÓN PRINCIPAL (TU CÓDIGO RESTAURADO)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    gc.collect()
    
    L, K, H = 30, 30, 2
    Tmax = 3.5
    Tmin = 0.1
    deltaT = 0.02
    pasos_medicion = 30000
    pasos_termalizacion = 200000 # Un poco más alto para asegurar
    max_workers = 8
    num_replicas = 30
    
    J_intra = 1.0
    J1 = 0.05
    J2 = 0.2
    J3 = 0.5
    
    start = time.time()
    
    # Ejecutamos las 3 simulaciones explícitamente como querías
    T, msp_1, msp_err1, msf_1, msf_err1, allcs_1, allcs_err1 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J_intra, J1)
    
    T, msp_2, msp_err2, msf_2, msf_err2, allcs_2, allcs_err2 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J_intra, J2)
    
    T, msp_3, msp_err3, msf_3, msf_err3, allcs_3, allcs_err3 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J_intra, J3)

    print(f"Tiempo total: {time.time() - start:.2f} s")

    # --- GRAFICAR ---

    # 1. Mp (Paramagnética)
    plt.figure(figsize=(6,4))
    plt.errorbar(T, msp_1, yerr=msp_err1, fmt='o', label=f'J_i = {J1}', alpha=0.7, markersize=3)
    plt.fill_between(T, msp_1 - msp_err1, msp_1 + msp_err1, alpha=0.18)

    plt.errorbar(T, msp_2, yerr=msp_err2, fmt='o', label=f'J_i = {J2}', alpha=0.7, markersize=3)
    plt.fill_between(T, msp_2 - msp_err2, msp_2 + msp_err2, alpha=0.18)

    plt.errorbar(T, msp_3, yerr=msp_err3, fmt='o', label=f'J_i = {J3}', alpha=0.7, markersize=3)
    plt.fill_between(T, msp_3 - msp_err3, msp_3 + msp_err3, alpha=0.18)

    plt.xlabel('kT/J')
    plt.ylabel('|Mz| Paramagnético')
    plt.title('Magnetización por sitio – capa paramagnética')
    plt.legend()
    plt.grid(True)
    plt.savefig("Mp_vs_T.png", dpi=200, bbox_inches="tight")
    plt.show()

    # 2. Mf (Ferromagnética)
    plt.figure(figsize=(6,4))
    plt.errorbar(T, msf_1, yerr=msf_err1, fmt='o', label=f'J_i = {J1}', alpha=0.7, markersize=3)
    plt.fill_between(T, msf_1 - msf_err1, msf_1 + msf_err1, alpha=0.18)

    plt.errorbar(T, msf_2, yerr=msf_err2, fmt='o', label=f'J_i = {J2}', alpha=0.7, markersize=3)
    plt.fill_between(T, msf_2 - msf_err2, msf_2 + msf_err2, alpha=0.18)

    plt.errorbar(T, msf_3, yerr=msf_err3, fmt='o', label=f'J_i = {J3}', alpha=0.7, markersize=3)
    plt.fill_between(T, msf_3 - msf_err3, msf_3 + msf_err3, alpha=0.18)

    plt.xlabel('kT/J')
    plt.ylabel('|Mz| Ferromagnético')
    plt.title('Magnetización por sitio – capa ferromagnética')
    plt.legend()
    plt.grid(True)
    plt.savefig("Mf_vs_T.png", dpi=200, bbox_inches="tight")
    plt.show()

    # 3. Fluctuaciones (C)
    plt.figure(figsize=(6,4))
    plt.errorbar(T, allcs_1, yerr=allcs_err1, fmt='o', label=f'J_i = {J1}', alpha=0.7, markersize=3)
    plt.fill_between(T, allcs_1 - allcs_err1, allcs_1 + allcs_err1, alpha=0.18)
    
    plt.errorbar(T, allcs_2, yerr=allcs_err2, fmt='o', label=f'J_i = {J2}', alpha=0.7, markersize=3)
    plt.fill_between(T, allcs_2 - allcs_err2, allcs_2 + allcs_err2, alpha=0.18)
    
    plt.errorbar(T, allcs_3, yerr=allcs_err3, fmt='o', label=f'J_i = {J3}', alpha=0.7, markersize=3)
    plt.fill_between(T, allcs_3 - allcs_err3, allcs_3 + allcs_err3, alpha=0.18)
    
    plt.xlabel('kT/J')
    plt.ylabel('Calor Específico (C)')
    plt.title('Fluctuaciones de la energía')
    plt.legend()
    plt.grid(True)
    plt.savefig("Fluctuaciones_E.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("Proceso terminado correctamente.")
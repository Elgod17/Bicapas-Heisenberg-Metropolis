'SIMULACION DE LA BICADA FERRO/PARAMAGNETICO BAJO LA ACCIÓN DE UN CAMPO MAGNETICO EXTERNO H'
'ESTA ES UNA ADAPTACIÓN HECHA POR MI AL CODIGO "bicapa_ferro_paramagnetico.py"'

import numpy as np
import random
import matplotlib
matplotlib.use("Agg") # No mostrar ventana
import matplotlib.pyplot as plt
from numba import njit, config as numba_config
import concurrent.futures
import multiprocessing
import gc
import time

# Configuración hilos
numba_config.THREADING_LAYER = 'workqueue'

# -----------------------------------------------------------------------------
# 1. MOTOR FÍSICO (NUMBA)
# -----------------------------------------------------------------------------

@njit
def random_spin():
    'ALGORITMO DE MORSAGLIA PARA MUESTREAR PUNTOS DE LA ESFEREA UNITARIA'
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
    config[1:l+1, 1:k+1, 1:h+1, :] = np.array([0.0, 0.0, 1.0])
    return config

@njit
def calcular_energia_total(cfg, l, k, h, J, J_interaccion, H):
    E = 0.0
    for x in range(1, l + 1):
        for y in range(1, k + 1):
            for z in range(1, h + 1):
                S = cfg[x, y, z]
                if x < l: E -= J * np.dot(S, cfg[x+1, y, z])
                if y < k: E -= J * np.dot(S, cfg[x, y+1, z])  
                if z < h: E -= J_interaccion * np.dot(S, cfg[x, y, z+1])
                E -= H * S[2]
    return E

@njit
def metropolis(config, T, pasos_term, pasos_med, l, k, h, J, J_interaccion,H):
    cfg = config
    
    # --- A. TERMALIZACIÓN (Loop explícito) ---
    for _ in range(pasos_term):
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
        campo += J_interaccion * vec_z + np.array([0.0,0.0,H]) 
        
        dE = -np.dot((S_new - S_old), campo)

        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[x,y,z] = S_new

    # --- B. MEDICIÓN (Loop explícito) ---
    E_sum = 0.0
    E2_sum = 0.0
    
    # Acumuladores
    mz_para_acc = 0.0   # |Mz|
    mod_para_acc = 0.0  # |Vector|
    mz_ferro_acc = 0.0
    mod_ferro_acc = 0.0
    
    n_medidas = 0
    E_actual = calcular_energia_total(cfg, l, k, h, J, J_interaccion, H)
    
    # Barrido
    volumen = l * k * h
    pasos_barrido = volumen 
    
    for step in range(pasos_med):
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
        
        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[x,y,z] = S_new
            E_actual += dE
            
        # TOMA DE DATOS (Cada Sweep)
        if step % pasos_barrido == 0:
            E_sum += E_actual
            E2_sum += E_actual**2
            
            # 1. Capa Para (z=1)
            vec_para = np.zeros(3)
            for i in range(1, l+1):
                for j in range(1, k+1):
                    vec_para += cfg[i, j, 1]
            
            N_capa = l * k
            m_para_inst = vec_para / N_capa
            mz_para_acc += np.abs(m_para_inst[2])
            mod_para_acc += np.linalg.norm(m_para_inst)
            
            # 2. Capa Ferro (z > 1)
            if h > 1:
                vec_ferro = np.zeros(3)
                for z_idx in range(2, h+1):
                    for i in range(1, l+1):
                        for j in range(1, k+1):
                            vec_ferro += cfg[i, j, z_idx]
                
                N_bulk = l * k * (h - 1)
                m_ferro_inst = vec_ferro / N_bulk
                mz_ferro_acc += np.abs(m_ferro_inst[2])
                mod_ferro_acc += np.linalg.norm(m_ferro_inst)
            
            n_medidas += 1

    # --- RESULTADOS ---
    if n_medidas > 0:
        res_mz_p = mz_para_acc / n_medidas
        res_mod_p = mod_para_acc / n_medidas
        res_mz_f = mz_ferro_acc / n_medidas
        res_mod_f = mod_ferro_acc / n_medidas
        
        E_avg = E_sum / n_medidas
        E2_avg = E2_sum / n_medidas
        var_E = E2_avg - (E_avg**2)
        if var_E < 0: var_E = 0.0
        c = var_E / (volumen * (T**2))
    else:
        res_mz_p, res_mod_p, res_mz_f, res_mod_f, c = 0.0, 0.0, 0.0, 0.0, 0.0

    return res_mz_p, res_mod_p, res_mz_f, res_mod_f, c

# -----------------------------------------------------------------------------
# 2. GESTIÓN
# -----------------------------------------------------------------------------

def m_vs_t(Ts, p_term, p_med, l, k, h, J, J_interaccion, H):
    arr_mz_p, arr_mod_p = [], []
    arr_mz_f, arr_mod_f = [], []
    arr_c = []
    
    curr_config = config_ordenada(l,k,h)
    
    for T in Ts:
        # AQUÍ PASAMOS LOS DOS PARÁMETROS SEPARADOS
        mz_p, mod_p, mz_f, mod_f, c = metropolis(curr_config, T, p_term, p_med, l, k, h, J, J_interaccion, H)
        
        arr_mz_p.append(mz_p)
        arr_mod_p.append(mod_p)
        arr_mz_f.append(mz_f)
        arr_mod_f.append(mod_f)
        arr_c.append(c)
        
    return (np.array(arr_mz_p), np.array(arr_mod_p), 
            np.array(arr_mz_f), np.array(arr_mod_f), 
            np.array(arr_c))

def simulaciones_mt(Tmax, Tmin, deltaT, p_term, p_med, max_workers, num_replicas, l, k, h, J, J_interaccion, H):
    Ts = np.arange(Tmin, Tmax + deltaT, deltaT)
    if Ts[0] < 1e-4: Ts = Ts[1:]
    
    print(f"--> J_int = {J_interaccion} | Term={p_term}, Med={p_med}...")
    
    rep_mz_p, rep_mod_p = [], []
    rep_mz_f, rep_mod_f = [], []
    rep_c = []
    
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Se envían p_term y p_med por separado
        futures = [executor.submit(m_vs_t, Ts, p_term, p_med, l, k, h, J, J_interaccion, H) for _ in range(num_replicas)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                r1, r2, r3, r4, r5 = future.result()
                rep_mz_p.append(r1)
                rep_mod_p.append(r2)
                rep_mz_f.append(r3)
                rep_mod_f.append(r4)
                rep_c.append(r5)
            except Exception as e:
                print(f"Error: {e}")

    # Promedios
    mat = np.array(rep_mz_p)
    mz_p_mean = np.mean(mat, axis=0)
    mz_p_err = np.std(mat, axis=0) / np.sqrt(num_replicas)
    
    mat = np.array(rep_mod_p)
    mod_p_mean = np.mean(mat, axis=0)
    mod_p_err = np.std(mat, axis=0) / np.sqrt(num_replicas)
    
    mat = np.array(rep_mz_f)
    mz_f_mean = np.mean(mat, axis=0)
    mz_f_err = np.std(mat, axis=0) / np.sqrt(num_replicas)
    
    mat = np.array(rep_mod_f)
    mod_f_mean = np.mean(mat, axis=0)
    mod_f_err = np.std(mat, axis=0) / np.sqrt(num_replicas)
    
    mat = np.array(rep_c)
    c_mean = np.mean(mat, axis=0)
    c_err = np.std(mat, axis=0) / np.sqrt(num_replicas)
    
    return Ts, mz_p_mean, mz_p_err, mod_p_mean, mod_p_err, mz_f_mean, mz_f_err, mod_f_mean, mod_f_err, c_mean, c_err

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    gc.collect()
    
    # --- PARÁMETROS ---
    #L, K, H = 100, 100, 2
    #Tmax = 1.5
    #Tmin = 0.1
    #deltaT = 0.02
    #pasos_medicion = 400000
    #pasos_termalizacion = 800000 # Un poco más alto para asegurar
    #max_workers = 8
    #num_replicas = 20
    #

    #L, K, H = 20, 20, 2
    #Tmax = 4.0
    #Tmin = 0.1
    #deltaT = 0.1
    #
    #pasos_medicion = 3000
    #pasos_termalizacion = 6000 
    #max_workers = 10
    #num_replicas = 3000 
    
    L, K, H = 30, 30, 2
    Tmax = 1.5
    Tmin = 0.1
    deltaT = 0.05
    pasos_medicion = 50000
    pasos_termalizacion = 500000 # Un poco más alto para asegurar
    max_workers = 10
    num_replicas = 300
    h = 0.0001




    J_intra = 1.0
    J_interacciones = [0.1] 
    
    start_time = time.time()
    data = {}

    for J_val in J_interacciones:
        data[J_val] = simulaciones_mt(
            Tmax, Tmin, deltaT, pasos_termalizacion, pasos_medicion, 
            max_workers, num_replicas, 
            L, K, H, J_intra, J_val, h
        )

    print(f"Listo patrón. Tiempo: {time.time() - start_time:.2f} s")

    # =========================================================================
    # GRAFICACIÓN (USANDO SOLO PASOS DE MEDICIÓN EN TITULOS)
    # =========================================================================
    
    # 1. MODULO PARA
    plt.figure(figsize=(8,6))
    for J_val in J_interacciones:
        Ts, _, _, mod_p, mod_p_e, _, _, _, _, _, _ = data[J_val]
        plt.errorbar(Ts, mod_p, yerr=mod_p_e, fmt='o-', label=f'J_int = {J_val}', capsize=3)
    
    plt.xlabel('Temperatura (kT/J)')
    plt.ylabel('|M| (Módulo)')
    plt.title(f'Módulo Magnetización - Capa Paramagnética\nPasos de MEDICIÓN: {pasos_medicion}') # <--- AQUÍ
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Modulo_Para_{pasos_medicion}medidas.png", dpi=200) # <--- AQUÍ
    plt.close()

    # 2. MZ PARA
    plt.figure(figsize=(8,6))
    for J_val in J_interacciones:
        Ts, mz_p, mz_p_e, _, _, _, _, _, _, _, _ = data[J_val]
        plt.errorbar(Ts, mz_p, yerr=mz_p_e, fmt='x--', label=f'J_int = {J_val}', capsize=3, alpha=0.7)
    
    plt.xlabel('Temperatura (kT/J)')
    plt.ylabel('|Mz|')
    plt.title(f'Magnetización Z - Capa Paramagnética\nPasos de MEDICIÓN: {pasos_medicion}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Mz_Para_{pasos_medicion}medidas.png", dpi=200)
    plt.close()

    # 3. MODULO FERRO
    plt.figure(figsize=(8,6))
    for J_val in J_interacciones:
        Ts, _, _, _, _, _, _, mod_f, mod_f_e, _, _ = data[J_val]
        plt.errorbar(Ts, mod_f, yerr=mod_f_e, fmt='s-', label=f'J_int = {J_val}', capsize=3)
    
    plt.xlabel('Temperatura (kT/J)')
    plt.ylabel('|M| (Módulo)')
    plt.title(f'Módulo Magnetización - Capa Ferromagnética\nPasos de MEDICIÓN: {pasos_medicion}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Modulo_Ferro_{pasos_medicion}medidas.png", dpi=200)
    plt.close()

    # 4. MZ FERRO
    plt.figure(figsize=(8,6))
    for J_val in J_interacciones:
        Ts, _, _, _, _, mz_f, mz_f_e, _, _, _, _ = data[J_val]
        plt.errorbar(Ts, mz_f, yerr=mz_f_e, fmt='^--', label=f'J_int = {J_val}', capsize=3, alpha=0.7)
    
    plt.xlabel('Temperatura (kT/J)')
    plt.ylabel('|Mz|')
    plt.title(f'Magnetización Z - Capa Ferromagnética\nPasos de MEDICIÓN: {pasos_medicion}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Mz_Ferro_{pasos_medicion}medidas.png", dpi=200)
    plt.close()

    # 5. CALOR ESPECIFICO
    plt.figure(figsize=(8,6))
    for J_val in J_interacciones:
        Ts, _, _, _, _, _, _, _, _, c, ce = data[J_val]
        plt.errorbar(Ts, c, yerr=ce, fmt='D-', label=f'J_int = {J_val}', capsize=3)
    
    plt.xlabel('Temperatura (kT/J)')
    plt.ylabel('Cv')
    plt.title(f'Calor Específico\nPasos de MEDICIÓN: {pasos_medicion}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Calor_Especifico_definitivo_medidas.png", dpi=200)
    plt.close()

    print("Todo separado y graficado con el número de mediciones como título. ¿Algo más, señor?")
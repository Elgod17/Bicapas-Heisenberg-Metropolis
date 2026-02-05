'ESTE ES EL CODIGO QUE USÉ INICIALMENTE PARA SIMULAR LA BICAPA PARAMAGNETICO/FERRO. '
'PERO ESTE CÓDIGO FALLA POR RAZONES QUE NO ENTIENDO (SE QUEDA CONGELADO DESPUES DE '
'TERMINAR LAS SIMULACIONES). SEGUN GEMINI 3, PODRÍA SER POR UN MAL USO DEL MÓDULO'
'DE MULTIPROCESSING AL ESTAR OCUPANDO DEMASIADOS HILOS.'
'LA VERSION CORREGIDA POR GEMINI 3 DE ESTE CODIGO SE TITULA "bicapa_ferro_paramagnetico.py".'
'SE USA MULTIPROCESSING EN ESTE Y LOS OTRSO CODIGOS PARA GENERAR VARIAS SIMULACIONES'
'SIMULTANEAMENTE'


import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, config as numba_config
import concurrent.futures
import multiprocessing
import gc
import time

# Configuración para evitar conflictos de hilos
numba_config.THREADING_LAYER = 'workqueue'


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
def config_ordenada(l, k, h):
    # Creamos array con bordes extra para condiciones de frontera
    config = np.zeros((l+2, k+2, h+2, 3))
    # Llenamos solo el interior (del 1 al l, del 1 al k)
    # Apuntando todos hacia arriba en Y (vector 0, 1, 0)
    config[1:l+1, 1:k+1, 1:k+1, :] = np.array([0.0, 0.0, 1.0])
    return config







@njit
def vecinos(cfg,x,y,z,h,J,J_interaccion):
    if z == h:
        veci = cfg[x, y, z-1]

    elif z == h-1:
        veci = J_interaccion * cfg[x,y,h+1] + J * (cfg[x,y,z-1] + cfg[x+1,y,z] + cfg[x-1,y,z] + cfg[x,y+1,z] + cfg[x,y-1,z])

    else:
        veci = J * (cfg[x,y,z+1] + cfg[x,y,z-1] + cfg[x+1,y,z] + cfg[x-1,y,z] + cfg[x,y+1,z] + cfg[x,y-1,z])

    return veci





# 3. Metrópolis optimizado con Numba
@njit
def metropolis(config, T, pasos_locales, l, k, h, J, J_interaccion):
    # No necesitamos copiar config si vamos a modificarlo in-place y devolver la magnetización
    # Pero si quieres no dañar el original fuera de la función, hacemos copy:
    cfg = config
    

    E = 0
    for x in range(1, l+1):
        for y in range(1, k+1):
            E = J * np.dot(cfg[x, y, 1],cfg[x+1, y+1, 1]) + J_interaccion * np.dot(cfg[x,y,1], cfg[x,y,2])


    for _ in range(int(pasos_locales/2)):
        # Elegir sitio al azar
        x = np.random.randint(1, l + 1)
        y = np.random.randint(1, k + 1)
        z = np.random.randint(1, h + 1)
        
        S_new = random_spin()
        S_old = cfg[x,y,z]
        
        # Suma vectorial de los 4 vecinos

        veci = vecinos(cfg,x,y,z,h,J,J_interaccion)
        
        # Cambio de energía: -(S_new - S_old) * Vecinos
        # dE = E_new - E_old
        # E = -S * Vecinos
        dE = -np.dot((S_new - S_old), veci)
        
        # Criterio de Metropolis
        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[x,y,z] = S_new
            E = E + dE





    e = []
    e2 = []
    e.append(E)
    e2.append(E**2)


    for _ in range(int(pasos_locales/2)):
        # Elegir sitio al azar
        x = np.random.randint(1, l + 1)
        y = np.random.randint(1, k + 1)
        z = np.random.randint(1, h + 1)
        
        S_new = random_spin()
        S_old = cfg[x,y,z]
        
        # Suma vectorial de los 4 vecinos

        veci = vecinos(cfg,x,y,z,h,J,J_interaccion)
        
        # Cambio de energía: -(S_new - S_old) * Vecinos
        # dE = E_new - E_old
        # E = -S * Vecinos
        dE = -np.dot((S_new - S_old), veci)
        
        # Criterio de Metropolis
        if dE < 0 or np.random.random() < np.exp(-dE/T):
            cfg[x,y,z] = S_new
            E = E + dE
            e.append(E)
            e2.append(E**2)

    # Calcular Magnetización Total del sistema (Promedio de la norma del vector suma)
    # OJO: Solo sumamos el interior, no los bordes (que son cero)
    

            
    # Retornamos la magnitud de la magnetización promedio por espín
    Mz_ferro = np.sum(config, axis = 1)
    Mz_para = np.sum(config, axis = 2)
    c = np.mean(np.array(e2)) - np.mean(np.array(e))**2
    return Mz_para, Mz_ferro, c












# 4. Loop de Temperaturas
def m_vs_t(Ts, pasos_termalizacion, pasos_medicion, l, k, h, J, J_interaccion):
    Ms_p = []
    Ms_f = []
    cs = []
    
    # Usamos la configuración actual para la siguiente temperatura (recocido)
    curr_config = config_ordenada(l,k,h)
    
    # 1. Termalización inicial (calentamiento) a la primera T
    print(f"Termalizando a T={Ts[0]}...")
    mz_p, mz_f, c = metropolis(curr_config, Ts[0], pasos_termalizacion, l, k, h, J, J_interaccion)
    Ms_p.append(mz_p)
    Ms_f.append(mz_f)
    cs.append(c)
    
    # 2. Barrido de temperaturas
    for i, T in enumerate(Ts[1:]):
        # Usamos la configuración resultante anterior como inicio de esta
        mz_p, mz_f, c = metropolis(curr_config, Ts[0], pasos_termalizacion, l, k, h, J, J_interaccion)
        Ms_p.append(mz_p)
        Ms_f.append(mz_f)
        cs.append(c)
        
        # Barra de progreso simple
        if i % 10 == 0:
            print(f"T = {T:.2f}")
            
    return Ms_p/np.max(Ms_p), Ms_f/np.max(Ms_f), cs/np.max(cs)





def simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, l, k, h, J, J_interaccion):
  Ts = np.arange(Tmin, Tmax + deltaT, deltaT)
  # Ts, pasos_termalizacion, pasos_medicion, l, k, h, J, J_interaccion
  args_replicas = [(Ts, pasos_termalizacion, pasos_medicion, l, k, h, J, J_interaccion) for _ in range(num_replicas)]
  msp = []
  msf = []
  allcs = []
  completadas = 0
  ctx = multiprocessing.get_context('spawn')
  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = {executor.submit(m_vs_t, *args): i for i, args in enumerate(args_replicas)}
      
      for future in concurrent.futures.as_completed(futures):
          replica_num = futures[future]
          try:
              mp,mf,cs = future.result()
              msp.append(mp)
              msf.append(msf)
              allcs.append(cs)

              completadas += 1
              if completadas % 5 == 0 or completadas == num_replicas:
                  print(f"    Progreso: {completadas}/{num_replicas} réplicas completadas")
          except Exception as e:
              print(f"    ✗ Réplica {replica_num+1} falló: {e}")
  msp_ = np.mean(np.array(msp), axis = 0)
  msf_ = np.mean(np.array(msf), axis = 0)
  allcs_ = np.mean(np.array(allcs), axis = 0)

  msp_err = np.std(np.array(msp), axis = 0) / np.sqrt(num_replicas)
  msf_err = np.std(np.array(msf), axis = 0) / np.sqrt(num_replicas)
  allcs_err = np.std(np.array(allcs), axis = 0) / np.sqrt(num_replicas)


  return Ts, msp_, msp_err, msf_, msf_err, allcs_, allcs_err
 




# --- EJECUCIÓN ---

# Parámetros

# Graficar
#plt.figure(figsize=(8, 5))
#plt.plot(Temperaturas, magnetizaciones, 'o-', markersize=4, label='Bicapa Heisenberg')
#plt.title(f"Magnetización vs Temperatura (Red {L}x{K})")
#plt.xlabel("Temperatura (T)")
#plt.ylabel("Magnetización Promedio |M|")
#plt.grid(True)
#plt.legend()
#plt.show()



if __name__ == "__main__":
    gc.collect()

    # PARAMENTROS DE LA SIMULACION
    L, K, H = 20,20,2  # ancho, largo y alto
    Tmax = 5
    Tmin = 0.0001
    deltaT = 0.1  # paso de la temperatura
    pasos_medicion = 2000
    pasos_termalizacion = 10000
    max_workers = 10     # numero de nucleos para simular simultaneamente
    num_replicas = 20
    J = 1
    J1 = 0.05
    J2 = 0.2
    J3 = 0.5
    Temperaturas = np.arange(0.1, 4.0, 0.04) # Rango típico para ver transición de fase
    
    
    # Correr simulación
    T, msp_1, msp_err1, msf_1, msf_err1, allcs_1, allcs_err1 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J1, J1)
    T, msp_2, msp_err2, msf_2, msf_err2, allcs_2, allcs_err2 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J2, J2)
    T, msp_3, msp_err3, msf_3, msf_err3, allcs_3, allcs_err3 = simulaciones_mt(Tmax, Tmin, deltaT, pasos_medicion, pasos_termalizacion, max_workers, num_replicas, L, K, H, J3, J3)



    plt.figure(figsize=(6,4))

    plt.scatter(T, msp_1, label=f'J_i = {J1}')
    plt.fill_between(T, msp_1 - msp_err1, msp_1 + msp_err1, alpha=0.18)

    plt.scatter(T, msp_2, label=f'J_i = {J2}')
    plt.fill_between(T, msp_2 - msp_err2, msp_2 + msp_err2, alpha=0.18)

    plt.scatter(T, msp_3, label=f'J_i = {J3}')
    plt.fill_between(T, msp_3 - msp_err3, msp_3 + msp_err3, alpha=0.18)

    plt.xlabel('kT/J')
    plt.ylabel('Mₚ normalizado')
    plt.title('Magnetización por sitio – capa paramagnética')
    plt.legend()
    plt.grid()

    plt.savefig("Mp_vs_T.png", dpi=200, bbox_inches="tight")
    plt.close()










    plt.figure(figsize=(6,4))

    plt.scatter(T, msf_1, label=f'J_i = {J1}')
    plt.fill_between(T, msf_1 - msf_err1, msf_1 + msf_err1, alpha=0.18)

    plt.scatter(T, msf_2, label=f'J_i = {J2}')
    plt.fill_between(T, msf_2 - msf_err2, msf_2 + msf_err2, alpha=0.18)

    plt.scatter(T, msf_3, label=f'J_i = {J3}')
    plt.fill_between(T, msf_3 - msf_err3, msf_3 + msf_err3, alpha=0.18)

    plt.xlabel('kT/J')
    plt.ylabel('M_f normalizado')
    plt.title('Magnetización por sitio – capa ferromagnética')
    plt.legend()
    plt.grid()

    plt.savefig("Mf_vs_T.png", dpi=200, bbox_inches="tight")
    plt.close()





    plt.figure(figsize=(6,4))

    plt.scatter(T, allcs_1, label=f'J_i = {J1}')
    plt.fill_between(T, allcs_1 - allcs_err1, allcs_1 + allcs_err1, alpha=0.18)

    plt.scatter(T, allcs_2, label=f'J_i = {J2}')
    plt.fill_between(T, allcs_2 - allcs_err2, allcs_2 + allcs_err2, alpha=0.18)

    plt.scatter(T, allcs_3, label=f'J_i = {J3}')
    plt.fill_between(T, allcs_3 - allcs_err3, allcs_3 + allcs_err3, alpha=0.18)

    plt.xlabel('kT/J')
    plt.ylabel(r'$(\Delta E)^2 / J^2$')
    plt.title('Fluctuaciones de la energía')
    plt.legend()
    plt.grid()

    plt.savefig("Fluctuaciones_E.png", dpi=200, bbox_inches="tight")
    plt.close()


#    ##### MAGNETIZACION CAPA PARAMAGNETICA
#    plt.figure()
#
#    plt.scatter(T, msp_1,label=f'J_i = {J1}')
#    plt.fill_between(msp_1 - msp_err1, msp_1 - msp_err1,alpha=0.18)
#
#    plt.scatter(T, msp_2,label=f'J_i = {J2}')
#    plt.fill_between(msp_2 - msp_err2, msp_2 - msp_err2,alpha=0.18)
#    
#    plt.scatter(T, msp_3,label=f'J_i = {J3}')
#    plt.fill_between(msp_3 - msp_err3, msp_3 - msp_err3,alpha=0.18)  
#
#    plt.xlabel('kT/J')
#    plt.ylabel('M_p normalizado')
#    plt.title('Magnetización por sitio en la capa paramagnética contra temperatura')
#    plt.legend()
#    plt.grid()
#    plt.show()
#
#
#
#
#
#    ##### MAGNETIZACION CAPA FERRO
#    plt.figure()
#
#    plt.scatter(T, msf_1,label=f'J_i = {J1}')
#    plt.fill_between(msf_1 - msf_err1, msf_1 - msf_err1,alpha=0.18)
#
#    plt.scatter(T, msp_2,label=f'J_i = {J2}')
#    plt.fill_between(msf_2 - msf_err2, msf_2 - msf_err2,alpha=0.18)
#    
#    plt.scatter(T, msf_3,label=f'J_i = {J3}')
#    plt.fill_between(msf_3 - msf_err3, msf_3 - msf_err3,alpha=0.18)  
#
#    plt.xlabel('kT/J')
#    plt.ylabel('M_f normalizado')
#    plt.title('Magnetización por sitio en la capa ferromagnética contra temperatura')
#    plt.legend()
#    plt.grid()
#    plt.show()
#
#
#
#
#
#
#    plt.figure()
#
#    plt.scatter(T, allcs_1,label=f'J_i = {J1}')
#    plt.fill_between(allcs_1 - allcs_err1, allcs_1 - allcs_err1,alpha=0.18)
#
#    plt.scatter(T, allcs_2,label=f'J_i = {J2}')
#    plt.fill_between(allcs_2 - allcs_err2, allcs_2 - allcs_err2,alpha=0.18)
#    
#    plt.scatter(T, allcs_3,label=f'J_i = {J3}')
#    plt.fill_between(allcs_3 - allcs_err3, allcs_3 - allcs_err3,alpha=0.18)  
#
#    plt.xlabel('kT/J')
#    plt.ylabel('(ΔE)²/J²')
#    plt.title('Fluctuaciones de la energía')
#    plt.legend()
#    plt.grid()
#    plt.show()
import numpy as np
from multiprocessing import Pool

def sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx):
    """
    Muestrea un subconjunto de partículas de 'start_idx' a 'end_idx'.
    """
    local_Ntot = end_idx - start_idx
    sigma = rho.sum(axis=0)  # Suma a lo largo del eje de 'theta'
    P = (sigma - sigma.min()) / sigma.ptp()

    philist = np.zeros(local_Ntot)
    rlist = np.zeros(local_Ntot)
    thetalist = np.zeros(local_Ntot)

    phi_area = (phi.max() - phi.min()) / (len(phi) - 1)
    r_area = (r.max() - r.min()) / (len(r) - 4)  # ajuste debido a [3:-4] al cargar 'r'

    N = start_idx
    while N < end_idx:
        _phi = np.random.uniform(phi.min(), phi.max())
        _r = np.random.uniform(r.min(), r.max())
        _theta = np.random.uniform(theta.min(), theta.max())
        
        iphi = min(int((_phi - phi.min()) / phi_area), len(phimed) - 1)
        ir = min(int((_r - r.min()) / r_area), len(rmed) - 1)
        itheta = np.random.randint(len(thetamed))

        _w = np.random.rand()
        
        if _w < P[ir, iphi]:
            philist[N - start_idx] = _phi
            rlist[N - start_idx] = _r
            thetalist[N - start_idx] = theta[itheta]
            N += 1

    return rlist, philist, thetalist

def parallel_sample_particles_3d(rho, phi, theta, rmed, phimed, thetamed, Ntot, num_processes):
    """
    Paraleliza el muestreo de partículas utilizando múltiples procesos.
    """
    pool = Pool(processes=num_processes)
    chunk_size = Ntot // num_processes
    results = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_processes - 1 else Ntot
        results.append(pool.apply_async(sample_particles_3d_partial, (rho, phi, theta, rmed, phimed, thetamed, start_idx, end_idx, Ntot)))
    
    pool.close()
    pool.join()

    # Combina los resultados de todos los procesos
    rlists, philists, thetalists = zip(*[result.get() for result in results])
    return np.concatenate(rlists), np.concatenate(philists), np.concatenate(thetalists)

def assign_particle_densities_from_grid_3d_partial(rho, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Asigna densidades a un subconjunto de partículas basadas en la densidad del grid 3D de FARGO3D.
    """
    local_Ntot = end_idx - start_idx
    particle_densities = np.zeros(local_Ntot)

    for idx in range(local_Ntot):
        r = rlist[idx]
        phi = philist[idx]
        theta = thetalist[idx]

        iphi = int((phi - phimed.min()) / phimed.ptp() * (len(phimed) - 1))
        ir = int((r - rmed.min()) / rmed.ptp() * (len(rmed) - 1))
        itheta = int((theta - thetamed.min()) / thetamed.ptp() * (len(thetamed) - 1))

        if 0 <= ir < len(rmed) and 0 <= iphi < len(phimed) and 0 <= itheta < len(thetamed):
            particle_densities[idx] = rho[itheta, ir, iphi]

    return particle_densities

def assign_particle_velocities_from_grid_3d_partial(vphi, vr, vtheta, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Asigna velocidades a un subconjunto de partículas basadas en la velocidad del grid 3D de FARGO3D.
    """
    print("=====| rlist.shape:  ", rlist.shape , "    |=====")
    print("=====| philist.shape: ", philist.shape, "    |=====")
    print("=====| thetalist.shape: ", thetalist.shape, " |=====")

    local_Ntot = (end_idx - start_idx)

    print("=====| start_idx:  ", start_idx , "    |=====")
    print("=====| end_idx: ", end_idx, "    |=====")
    print("=====| local_Ntot: ", local_Ntot, " |=====")

    particle_vphi = np.zeros(local_Ntot)
    particle_vr = np.zeros(local_Ntot)
    particle_vtheta = np.zeros(local_Ntot)

    for idx in range(local_Ntot):
        #print("=====| idx: ", idx, " |=====")
        r = rlist[idx]
        phi = philist[idx]
        theta = thetalist[idx]

        iphi = int((phi - phimed.min()) / phimed.ptp() * (len(phimed) - 1))
        ir = int((r - rmed.min()) / rmed.ptp() * (len(rmed) - 1))
        itheta = int((theta - thetamed.min()) / thetamed.ptp() * (len(thetamed) - 1))

        if 0 <= ir < len(rmed) and 0 <= iphi < len(phimed) and 0 <= itheta < len(thetamed):
            particle_vphi[idx] = vphi[itheta, ir, iphi]
            particle_vr[idx] = vr[itheta, ir, iphi]
            particle_vtheta[idx] = vtheta[itheta, ir, iphi]

    return particle_vphi, particle_vr, particle_vtheta

def assign_particle_internal_energies_from_grid_3d_partial(rlist, philist, thetalist, grid_energies, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Asigna energía interna a un subconjunto de partículas basadas en la energía interna del grid 3D de FARGO3D.
    """
    local_Ntot = end_idx - start_idx
    particle_energies = np.zeros(local_Ntot)

    for idx in range(local_Ntot):
        r = rlist[idx]
        phi = philist[idx]
        theta = thetalist[idx]

        iphi = int((phi - phimed.min()) / phimed.ptp() * (len(phimed) - 1))
        ir = int((r - rmed.min()) / rmed.ptp() * (len(rmed) - 1))
        itheta = int((theta - thetamed.min()) / thetamed.ptp() * (len(thetamed) - 1))

        if 0 <= ir < len(rmed) and 0 <= iphi < len(phimed) and 0 <= itheta < len(thetamed):
            particle_energies[idx] = grid_energies[itheta, ir, iphi]

    return particle_energies


############################################################################################################
############################################################################################################
######                                      SAMPLING TRILINEAL                                      ########
############################################################################################################
############################################################################################################


def sample_particles_3d_vectorized(rho, grid, Ntot, oversampling_factor=2):
    # Estimar cuántos puntos generar (puede necesitar ajustes)
    num_candidates = int(Ntot * oversampling_factor)

    # Generar puntos candidatos
    _rs = np.random.uniform(grid[0].min(), grid[0].max(), num_candidates)
    _phis = np.random.uniform(grid[1].min(), grid[1].max(), num_candidates)
    _thetas = np.random.uniform(grid[2].min(), grid[2].max(), num_candidates)

    # Interpolación trilineal para todos los puntos
    interpolated_densities = trilinear_interpolation(_rs, _phis, _thetas, grid, rho)

    # Filtrar puntos
    acceptance_probability = interpolated_densities / rho.max()
    accepted_indices = np.random.rand(num_candidates) < acceptance_probability

    # Asegurarse de seleccionar exactamente Ntot partículas
    accepted_r = _rs[accepted_indices][:Ntot]
    accepted_phi = _phis[accepted_indices][:Ntot]
    accepted_theta = _thetas[accepted_indices][:Ntot]

    return accepted_r, accepted_phi, accepted_theta

def trilinear_interpolation(point, grid, values):
    # Encuentra los índices de los puntos de la cuadrícula que rodean el punto de interés
    r_idx, phi_idx, theta_idx = [
        max(min(np.searchsorted(grid[dim], point[dim]) - 1, len(grid[dim]) - 2), 0)
        for dim in range(3)
    ]

    # Calcula los factores de interpolación para cada dimensión
    dr, dphi, dtheta = [
        (point[dim] - grid[dim][idx]) / (grid[dim][idx + 1] - grid[dim][idx])
        if grid[dim][idx + 1] > grid[dim][idx] else 0
        for dim, idx in enumerate([r_idx, phi_idx, theta_idx])
    ]

    # Interpola la densidad
    interpolated_value = 0
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                weight = ((dr if i == 1 else 1 - dr) *
                          (dphi if j == 1 else 1 - dphi) *
                          (dtheta if k == 1 else 1 - dtheta))
                interpolated_value += weight * values[theta_idx + k, r_idx + i, phi_idx + j]
    return interpolated_value

def sample_particles_3d_interpolated(rho, rmed, phimed, thetamed, start_idx, end_idx):
    local_Ntot = end_idx - start_idx
    rlist = np.zeros(local_Ntot)
    philist = np.zeros(local_Ntot)
    thetalist = np.zeros(local_Ntot)

    # Crear una cuadrícula para la interpolación
    grid = [rmed, phimed, thetamed]

    N = 0
    while N < local_Ntot:
        _r = np.random.uniform(rmed.min(), rmed.max())
        _phi = np.random.uniform(phimed.min(), phimed.max())
        _theta = np.random.uniform(thetamed.min(), thetamed.max())

        # Punto para interpolación
        point = [_r, _phi, _theta]

        # Realizar interpolación trilineal
        interpolated_density = trilinear_interpolation(point, grid, rho)

        # Decidir si aceptar la partícula basado en la densidad interpolada
        if np.random.rand() < interpolated_density / rho.max():
            rlist[N] = _r
            philist[N] = _phi
            thetalist[N] = _theta
            N += 1

    return rlist, philist, thetalist

import numpy as np
from multiprocessing import Pool

def sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx):
    """
    Samples a subset of particles from 'start_idx' to 'end_idx' using a rejection sampling method
    based on the local density distribution from a 3D grid.

    This function is designed to be used in a parallel processing context where the sampling
    of particles is distributed across multiple processes.

    :param rho: 3D density grid from simulation data.
    :param phi, theta: Angular coordinates from the simulation grid.
    :param rmed, phimed, thetamed: Median values of the radial, azimuthal, and polar angles.
    :param r: Radial coordinate array.
    :param start_idx, end_idx: Start and end indices for particle sampling in this subset.
    :return: Arrays of sampled r, phi, and theta values for the particles.
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

def assign_particle_densities_from_grid_3d_partial(rho, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Assigns densities to a subset of particles based on the 3D density grid from FARGO3D.

    :param rho: 3D density grid from the simulation.
    :param rlist, philist, thetalist: Lists of radial, azimuthal, and polar coordinates of particles.
    :param rmed, phimed, thetamed: Median values for the simulation grid coordinates.
    :param start_idx, end_idx: Start and end indices for processing this subset of particles.
    :return: Array of densities assigned to each particle in the subset.
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

    print("Density sampling completed.")

    return particle_densities

def assign_particle_velocities_from_grid_3d_partial(vphi, vr, vtheta, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Assigns velocities to a subset of particles based on the 3D velocity grid from FARGO3D.

    :param vphi, vr, vtheta: 3D arrays of azimuthal, radial, and polar velocities from the simulation.
    :param rlist, philist, thetalist: Lists of coordinates for particles.
    :param rmed, phimed, thetamed: Median values for the grid coordinates.
    :param start_idx, end_idx: Indices for the subset of particles to process.
    :return: Arrays of azimuthal, radial, and polar velocities for each particle in the subset.
    """
    local_Ntot = (end_idx - start_idx)
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
    Assigns internal energies to a subset of particles based on the 3D internal energy grid from FARGO3D.

    This function maps the internal energies from the grid to the particle representation, considering only a specific range of particles (defined by start_idx and end_idx) for efficient parallel processing.

    :param rlist, philist, thetalist: Lists of coordinates for particles.
    :param grid_energies: 3D grid of internal energies from the simulation.
    :param rmed, phimed, thetamed: Median values for the grid coordinates.
    :param start_idx, end_idx: Indices for the subset of particles to process.
    :return: Array of internal energies assigned to each particle in the subset.
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

    print("Internal energy sampling completed.")

    return particle_energies



############################################################################################################
######                                      SAMPLING TRILINEAL                                      ########
############################################################################################################

# on develop by the moment (2021-09-30)
def sample_particles_3d_vectorized(rho, grid, Ntot, oversampling_factor=2):
    """
    Samples particles using a vectorized approach with trilinear interpolation.

    This method uses oversampling and filtering based on interpolated densities to efficiently and accurately sample particles in the simulation domain.

    :param rho: 3D density grid from simulation data.
    :param grid: Grid arrays for r, phi, and theta coordinates.
    :param Ntot: Total number of particles to sample.
    :param oversampling_factor: Factor to control the initial oversampling rate.
    :return: Arrays of sampled r, phi, and theta coordinates for the particles.
    """
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
    """
    Performs trilinear interpolation for a given point within a 3D grid.

    :param point: The point (r, phi, theta) where interpolation is to be performed.
    :param grid: Grid arrays for r, phi, and theta coordinates.
    :param values: 3D array of values corresponding to the grid points.
    :return: Interpolated value at the given point.
    """
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


#revisar esta función
def sample_particles_3d_interpolated(rho, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Samples particles using trilinear interpolation within a specific range defined by start_idx and end_idx.

    This function is suitable for parallel processing where the sampling task is divided among multiple processors.

    :param rho: 3D density grid from simulation data.
    :param rmed, phimed, thetamed: Median values for the grid coordinates.
    :param start_idx, end_idx: Indices defining the range of particles to sample.
    :return: Arrays of sampled r, phi, and theta coordinates for the particles in the specified range.
    """
    local_Ntot = end_idx - start_idx
    rlist = np.zeros(local_Ntot)
    philist = np.zeros(local_Ntot)
    thetalist = np.zeros(local_Ntot)

    # Crear una cuadrícula para la interpolación [aquiiii]
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

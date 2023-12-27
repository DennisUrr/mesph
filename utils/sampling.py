import numpy as np

from utils.conversions import spherical_to_cartesian, to_cartesian

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
    theta_area = (theta.max() - theta.min()) / (len(theta) - 1)

    N = start_idx
    while N < end_idx:
        _phi = np.random.uniform(phi.min(), phi.max())
        _r = np.random.uniform(r.min(), r.max())
        _theta = np.random.uniform(theta.min(), theta.max())
        
        iphi = min(int((_phi - phi.min()) / phi_area), len(phimed) - 1)
        ir = min(int((_r - r.min()) / r_area), len(rmed) - 1)
        itheta = min(int((_theta - theta.min()) / theta_area), len(thetamed) - 1)
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

    # print("Density sampling completed.")

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

    # print("Internal energy sampling completed.")

    return particle_energies



############################################################################################################
######                                      SAMPLING TRILINEAL                                      ########
############################################################################################################

def sample_from_density_grid(rho, r, phi, theta, start_idx, end_idx):
    try:
        num_particles = end_idx - start_idx
        
        # Aplanar rho para obtener un array 1D
        rho_flat = rho.flatten()

        # Generar probabilidades a partir de rho
        probabilities = rho_flat / np.sum(rho_flat)

        # Generar índices de muestreo basados en estas probabilidades
        sampled_indices = np.random.choice(len(rho_flat), size=num_particles, p=probabilities)

        # Convertir índices 1D a índices 3D
        idx_theta, idx_r, idx_phi = np.unravel_index(sampled_indices, rho.shape)

        idx_theta = np.clip(idx_theta, 0, len(theta) - 1)
        idx_phi = np.clip(idx_phi, 0, len(phi) - 1)
        
        # Asignar coordenadas esféricas basadas en índices
        sampled_r = r[idx_r]
        sampled_phi = phi[idx_phi]  # Ajustar índices para phi
        sampled_theta = theta[idx_theta]  # Ajustar índices para theta

        return sampled_r, sampled_phi, sampled_theta
    except Exception as e:
        print("Error en sample_from_density_grid:", e)
        return None, None, None
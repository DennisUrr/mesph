import numpy as np
from scipy.interpolate import RegularGridInterpolator
import numba

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

def trilinear_interpolation_spherical(rho, r, phi, theta, points):
    """
    Realiza una interpolación trilineal en un grid regular en coordenadas esféricas.

    :param rho: Grid 3D de densidades.
    :param r: Array 1D de valores radiales.
    :param phi: Array 1D de valores azimutales.
    :param theta: Array 1D de valores polares.
    :param points: Puntos donde se desea interpolar. Cada punto debe estar en formato (r, phi, theta).
    :return: Valores interpolados en los puntos dados.
    """
    try:
        # Convertir rho a escala logarítmica
        rho_log = np.log10(rho)

        # Crea un interpolador en escala logarítmica
        interpolator = RegularGridInterpolator((theta, r, phi), rho_log)

        # Ajusta los puntos para la interpolación
        points_adjusted = np.array([(_theta, _r, _phi) for _theta, _r, _phi in points])

        # Realiza la interpolación en escala logarítmica
        interpolated_log_density = interpolator(points_adjusted)

        # Convierte de vuelta a la escala original
        return 10**interpolated_log_density
    except Exception as e:
        print(f"Error interpolating: {e}")
        raise e


@numba.njit
def sample_particles_3d_trilineal_partial(rho, phi, theta, r, start_idx, end_idx):

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
    try:
        local_Ntot = end_idx - start_idx

        philist = np.zeros(local_Ntot)
        rlist = np.zeros(local_Ntot)
        thetalist = np.zeros(local_Ntot)

        N = start_idx
        while N < end_idx:
            #print(theta.max(), theta[-2], theta[-1])
            _phi = np.random.uniform(phi.min(), phi[-2])
            _r = np.random.uniform(r.min(), r.max())
            _theta = np.random.uniform(theta.min(), theta[-2])
            #print(f"Random point: ({_phi}, {_r}, {_theta})")
            # Calcula la densidad interpolada en el punto seleccionado
            interpolated_density = trilinear_interpolation_spherical(rho, r, phi[:-1], theta[:-1], [(_theta, _r, _phi)])
            #print(f"Interpolated density: {interpolated_density}")
            #print(type(interpolated_density))
            # Decide si aceptar o rechazar la muestra basada en la densidad interpolada
            _w = np.random.rand()
            #_w = np.log10(np.random.rand())

            #print(f"Random number: {_w}")
            #print(f"Interpolated density: {interpolated_density[0]}")
            if _w < interpolated_density[0]:
                #print("Sample accepted.")
                philist[N - start_idx] = _phi
                rlist[N - start_idx] = _r
                thetalist[N - start_idx] = _theta
                N += 1

        return rlist, philist, thetalist
    except Exception as e:
        print(f"Error sampling particles: {e}")
        raise e
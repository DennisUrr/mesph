import numpy as np
from scipy.interpolate import RegularGridInterpolator
import numba
import traceback


@numba.njit
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

    #area in the face of the plane
    phi_area = (phi.max() - phi.min()) / (len(phi) - 1)
    r_area = (r.max() - r.min()) / (len(r) - 4)  # ajuste debido a [3:-4] al cargar 'r'
    theta_area = (theta.max() - theta.min()) / (len(theta) - 1)

    N = start_idx
    while N < end_idx:
        _phi = np.random.uniform(phi.min(), phi.max())
        _r = np.random.uniform(r.min(), r.max())
        _theta = np.random.uniform(theta.min(), theta.max())
        
        # representative point of the cell 
        iphi = min(int((_phi - phi.min()) / phi_area), len(phimed) - 1)
        ir = min(int((_r - r.min()) / r_area), len(rmed) - 4)
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

#@numba.njit
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

        # Calcula la densidad interpolada en el punto seleccionado
        interpolated_density = trilinear(rho, r, phi[:-1], theta[:-1], [(_theta, _r, _phi)])
        # Decide si aceptar o rechazar la muestra basada en la densidad interpolada
        _w = np.random.rand()

        if _w < interpolated_density[0]:
            #print("Sample accepted.")
            philist[N - start_idx] = _phi
            rlist[N - start_idx] = _r
            thetalist[N - start_idx] = _theta
            N += 1

    return rlist, philist, thetalist
    
    
@numba.njit
def sample_particles_3d_partial_1(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx):

    local_Ntot = end_idx - start_idx

    # Crear una matriz de probabilidad 3D
    P_3D = rho / rho.sum()

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
        ir = min(int((_r - r.min()) / r_area), len(rmed) - 4)
        itheta = min(int((_theta - theta.min()) / theta_area), len(thetamed) - 1)

        _w = np.random.rand()

        # Muestreo basado en la probabilidad 3D
        if _w < P_3D[itheta, ir, iphi]:
            #print("Sample accepted.")
            philist[N - start_idx] = _phi
            rlist[N - start_idx] = _r
            thetalist[N - start_idx] = _theta
            N += 1

    return rlist, philist, thetalist


#@numba.njit
def get_local_average(array, itheta, ir, iphi):
    # Calcula el promedio de una propiedad en una vecindad local
    theta_start, theta_end = max(itheta-1, 0), min(itheta+2, array.shape[0])
    r_start, r_end = max(ir-1, 0), min(ir+2, array.shape[1])
    phi_start, phi_end = max(iphi-1, 0), min(iphi+2, array.shape[2])
    
    return np.mean(array[theta_start:theta_end, r_start:r_end, phi_start:phi_end])
    

#@numba.njit
def sample_particles_3d_with_velocity_density(rho, phi, theta, rmed, phimed, thetamed, r, vphi, vr, vtheta, start_idx, end_idx):
    try:
        local_Ntot = end_idx - start_idx
        sigma = rho.sum(axis=0)  # Suma a lo largo del eje de 'theta'
        P = (sigma - sigma.min()) / sigma.ptp()

        philist = np.zeros(local_Ntot)
        rlist = np.zeros(local_Ntot)
        thetalist = np.zeros(local_Ntot)
        
        vphilist = np.zeros(local_Ntot)
        vrlist = np.zeros(local_Ntot)
        vthetalist = np.zeros(local_Ntot)

        rholist = np.zeros(local_Ntot)

        #area in the face of the plane
        phi_area = (phi.max() - phi.min()) / (len(phi) - 1)
        r_area = (r.max() - r.min()) / (len(r) - 4)  # ajuste debido a [3:-4] al cargar 'r'
        theta_area = (theta.max() - theta.min()) / (len(theta) - 1)

        N = start_idx
        while N < end_idx:
            _phi = np.random.uniform(phi.min(), phi.max())
            _r = np.random.uniform(r.min(), r.max())
            _theta = np.random.uniform(theta.min(), theta.max())
            
            # representative point of the cell 
            iphi = min(int((_phi - phi.min()) / phi_area), len(phimed) - 1)
            ir = min(int((_r - r.min()) / r_area), len(rmed) - 4)
            itheta = min(int((_theta - theta.min()) / theta_area), len(thetamed) - 1)
            _w = np.random.rand()
            
            if _w < P[ir, iphi]:
                philist[N - start_idx] = _phi
                rlist[N - start_idx] = _r
                thetalist[N - start_idx] = theta[itheta]
                
                # Ahora sampleamos las velocidades
                # Aquí usamos los índices ir, iphi, itheta para obtener las velocidades
                # Podemos hacer un promedio simple o ponderado de las velocidades en los puntos cercanos
                vphilist[N - start_idx] = get_local_average(vphi, itheta, ir, iphi)
                vrlist[N - start_idx] = get_local_average(vr, itheta, ir, iphi)
                vthetalist[N - start_idx] = get_local_average(vtheta, itheta, ir, iphi)
                rholist[N - start_idx] = get_local_average(rho, itheta, ir, iphi)
                N += 1

        return rlist, philist, thetalist, vrlist, vphilist, vthetalist, rholist
    except Exception as e:
        print(f"Error sampling velocities: {e}")
        traceback.print_exc()
        raise e



def interpolate_velocities(vr, vphi, vtheta, r, phi, theta, r_particles, phi_particles, theta_particles, FRAME="F", OMEGAFRAME=0):
    """
    Interpolates the velocities at the given particle positions using trilinear interpolation
    in spherical coordinates, accounting for the extra point in theta and phi.

    Parameters:
    vr, vphi, vtheta : 3D numpy arrays of the velocity components in spherical coordinates (r, phi, theta).
    r, phi, theta : 1D numpy arrays representing the grid points in spherical coordinates, with theta and phi
                    possibly including an extra point for the domain closure.
    r_particles, phi_particles, theta_particles : 1D numpy arrays of the particle positions in spherical coordinates.

    Returns:
    vr_interp, vphi_interp, vtheta_interp : 1D numpy arrays of interpolated velocities at particle positions.
    """
    # Adjust theta and phi to meet the requirements of the interpolator
    # Removing the extra point for theta and phi to match the velocity dimensions
    theta_adjusted = theta[:-1]
    phi_adjusted = phi[:-1]
    
    # Prepare the grid points for interpolation
    grid_for_interpolation = (theta_adjusted, r, phi_adjusted)

    # Create interpolators for each velocity component
    vr_interpolator = RegularGridInterpolator(grid_for_interpolation, vr, bounds_error=False, fill_value=None)
    vphi_interpolator = RegularGridInterpolator(grid_for_interpolation, vphi, bounds_error=False, fill_value=None)
    vtheta_interpolator = RegularGridInterpolator(grid_for_interpolation, vtheta, bounds_error=False, fill_value=None)

    # Ensure the phi values of particles are within the correct range
    phi_particles_adjusted = np.mod(phi_particles, 2 * np.pi)

    # Prepare particle positions for interpolation
    particle_positions = np.vstack((theta_particles, r_particles, phi_particles_adjusted)).T

    # Interpolate velocities at the particles' positions
    vr_interp = vr_interpolator(particle_positions)
    vphi_interp = vphi_interpolator(particle_positions)
    vtheta_interp = vtheta_interpolator(particle_positions)

    if FRAME == "G":
        # Asume que r_particles es un array de las distancias radiales de las partículas
        vphi_interp += r_particles * OMEGAFRAME

    return vr_interp, vphi_interp, vtheta_interp

from scipy.interpolate import RegularGridInterpolator
import numpy as np

def interpolate_densities(rho, r, phi, theta, r_particles, phi_particles, theta_particles):
    """
    Interpolates the densities at the given particle positions using trilinear interpolation
    in spherical coordinates.

    Parameters:
    rho : 3D numpy array of the density in spherical coordinates (r, phi, theta).
    r, phi, theta : 1D numpy arrays representing the grid points in spherical coordinates, with theta and phi
                    possibly including an extra point for the domain closure.
    r_particles, phi_particles, theta_particles : 1D numpy arrays of the particle positions in spherical coordinates.

    Returns:
    rho_interp : 1D numpy array of interpolated densities at particle positions.
    """
    # Adjust theta and phi to meet the requirements of the interpolator
    # Removing the extra point for theta and phi to match the density dimensions
    theta_adjusted = theta[:-1]
    phi_adjusted = phi[:-1]

    # Prepare the grid points for interpolation
    grid_for_interpolation = (theta_adjusted, r, phi_adjusted)

    # Create interpolator for the density
    rho_interpolator = RegularGridInterpolator(grid_for_interpolation, rho, bounds_error=False, fill_value=None)

    # Ensure the phi values of particles are within the correct range
    phi_particles_adjusted = np.mod(phi_particles, 2 * np.pi)

    # Prepare particle positions for interpolation
    particle_positions = np.vstack((theta_particles, r_particles, phi_particles_adjusted)).T

    # Interpolate densities at the particles' positions
    rho_interp = rho_interpolator(particle_positions)

    rho_interp = np.maximum(rho_interp, 1e-05)
    return rho_interp

def interpolate_internal_energies(grid_energies, r, phi, theta, r_particles, phi_particles, theta_particles):
    """
    Interpolates internal energies at the given particle positions using trilinear interpolation
    in spherical coordinates.

    Parameters:
    - grid_energies: 3D numpy array of the internal energy in spherical coordinates (r, phi, theta).
    - r, phi, theta: 1D numpy arrays representing the grid points in spherical coordinates, with theta and phi
                      possibly including an extra point for the domain closure.
    - r_particles, phi_particles, theta_particles: 1D numpy arrays of the particle positions in spherical coordinates.

    Returns:
    - energies_interp: 1D numpy array of interpolated internal energies at particle positions.
    """

    # Adjust theta and phi to meet the requirements of the interpolator
    # Removing the extra point for theta and phi to match the internal energy grid dimensions
    theta_adjusted = theta[:-1]
    phi_adjusted = phi[:-1]

    # Prepare the grid points for interpolation
    grid_for_interpolation = (theta_adjusted, r, phi_adjusted)

    # Create an interpolator for the internal energy grid
    energy_interpolator = RegularGridInterpolator(grid_for_interpolation, grid_energies, bounds_error=False, fill_value=None)

    # Ensure the phi values of particles are within the correct range
    phi_particles_adjusted = np.mod(phi_particles, 2 * np.pi)

    # Prepare particle positions for interpolation
    particle_positions = np.vstack((theta_particles, r_particles, phi_particles_adjusted)).T

    # Interpolate internal energies at the particles' positions
    energies_interp = energy_interpolator(particle_positions)

    return energies_interp
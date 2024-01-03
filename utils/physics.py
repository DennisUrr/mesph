import numpy as np
import traceback
from utils.sph_utils import gradient_gaussian_kernel, gradient_gaussian_kernel_vectorized


def compute_thickness(r, ASPECTRATIO):
    """
    Compute the thickness of the disk at a given radius r.

    :param r: Radius at which the disk's thickness is calculated.
    :param ASPECTRATIO: Aspect ratio of the disk.
    :return: Thickness of the disk at the given radius.
    """
    return ASPECTRATIO * r

def differential_volume_3d(r, ASPECTRATIO):
    """
    Compute the differential volume in 3D cylindrical coordinates.

    :param r: Radius in cylindrical coordinates.
    :param ASPECTRATIO: Aspect ratio of the disk.
    :return: Differential volume element in 3D cylindrical coordinates.
    """
    H = compute_thickness(r, ASPECTRATIO)
    # Elemento de volumen en coordenadas cilíndricas (r, phi, z)
    return r * H * 2 * np.pi


#REVISAR LA PARALELIZACIÓN DE ESTO
def total_disk_mass_3d(rho, r, phi, ASPECTRATIO):
    """
    Compute the total mass of the disk in 3D by integrating over the provided density grid.

    :param rho: 3D density grid.
    :param r: Radial coordinates grid.
    :param phi: Azimuthal coordinates grid.
    :param ASPECTRATIO: Aspect ratio of the disk.
    :return: Total mass of the disk.
    """
    total_mass = 0
    for i in range(len(r) - 1):
        for j in range(len(phi) - 1):
            # Calcula el volumen diferencial
            r_mid = (r[i] + r[i + 1]) / 2
            vol = differential_volume_3d(r_mid, ASPECTRATIO) * (phi[j + 1] - phi[j])
            # Suma la contribución de la masa
            total_mass += np.mean(rho[j, i, :]) * vol
    
    return total_mass

def compute_particle_mass_3d(nr, nphi, rho, ASPECTRATIO, params, Ntot=10000):
    """
    Compute the mass of each particle in 3D, assuming a uniform distribution of particles in the disk.

    :param nr: Number of radial divisions in the grid.
    :param nphi: Number of azimuthal divisions in the grid.
    :param rho: 3D density grid.
    :param ASPECTRATIO: Aspect ratio of the disk.
    :param params: Parameter dictionary containing simulation limits.
    :param Ntot: Total number of particles.
    :return: Mass of each particle.
    """
    if params is None:
        raise ValueError("params dictionary is required")

    r_max = float(params['YMAX'])
    r_min = float(params['YMIN'])
    phi_max = 2 * np.pi  # Asumiendo un disco completo

    r = np.linspace(r_min, r_max, nr)
    phi = np.linspace(0, phi_max, nphi)

    total_mass = total_disk_mass_3d(rho, r, phi, ASPECTRATIO)
    return total_mass / Ntot


def compute_pressure(densities, internal_energies, gamma):
    """
    Compute the pressure for each particle using the ideal gas law.

    :param densities: Array of densities for each particle.
    :param internal_energies: Array of internal energies for each particle.
    :param gamma: Adiabatic index.
    :return: Array of pressures for each particle.
    """
    return (gamma-1) * internal_energies * densities

# @timed
def compute_speed_sound(internal_energies, gamma):
    """
    Compute the speed of sound for each particle.

    :param internal_energies: Array of internal energies for each particle.
    :param gamma: Adiabatic index.
    :return: Array of sound speeds for each particle.
    """
    cs_square = (gamma - 1) * internal_energies
    cs = np.sqrt(cs_square)
    return cs

#@numba.jit
def compute_artificial_viscosity_3d_partial(positions, vx, vy, vz, densities, internal_energies, h_values, alpha=1, beta=2):
    """
    Calculates artificial viscosity for a subset of particles in 3D.

    :param positions: Positions of particles (Nx3 numpy array).
    :param vx, vy, vz: Velocities of particles in x, y, z directions (numpy arrays).
    :param densities: Densities of particles (numpy array).
    :param internal_energies: Internal energies of particles (numpy array).
    :param h_values: Smoothing lengths of particles (numpy array).
    :param alpha, beta: Parameters for artificial viscosity.
    :return: Artificial viscosity values for each particle (numpy array).
    """
    try:
        N = positions.shape[0]

        # Inicializa el array de viscosidad
        pi_c_i = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dz = positions[i, 2] - positions[j, 2]
                    dvx = vx[i] - vx[j]
                    dvy = vy[i] - vy[j]
                    dvz = vz[i] - vz[j]

                    r_ij = np.sqrt(dx**2 + dy**2 + dz**2)

                    c_ij = 0.5 * (np.sqrt(internal_energies[i]) + np.sqrt(internal_energies[j]))
                    h_ij = 0.5 * (h_values[i] + h_values[j])
                    condition = dvx*dx + dvy*dy + dvz*dz
                    mu_ij = h_ij * condition / (r_ij**2 + 1e-5 * h_ij**2)
                    rho_ij = 0.5 * (densities[i] + densities[j])

                    if condition < 0:
                        pi_c_ij = -alpha * c_ij * mu_ij / rho_ij + beta * mu_ij**2
                        pi_c_i[i] += pi_c_ij

        return pi_c_i
    except Exception as e:
        print("Error en compute_artificial_viscosity_3d_partial:", e)
        return None
#@numba.jit
def compute_acceleration_3d_partial(positions, densities, pressures, mass, h_values, viscosities):
    """
    Computes acceleration for a subset of particles in 3D.

    :param positions: Positions of particles (Nx3 numpy array).
    :param densities: Densities of particles (numpy array).
    :param pressures: Pressures of particles (numpy array).
    :param mass: Mass of each particle (scalar).
    :param h_values: Smoothing lengths of particles (numpy array).
    :param viscosities: Viscosity values for particles (numpy array).
    :return: Acceleration vectors for each particle (Nx3 numpy array).
    """
    try:
        N = positions.shape[0]
        # Inicializa los arrays de aceleración
        acceleration_x = np.zeros(N)
        acceleration_y = np.zeros(N)
        acceleration_z = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dz = positions[i, 2] - positions[j, 2]

                    r = np.sqrt(dx**2 + dy**2 + dz**2)

                    if r > 0:
                        grad_w = gradient_gaussian_kernel(r, 0.5 * (h_values[i] + h_values[j]))

                        term1 = pressures[i] / densities[i]**2
                        term2 = pressures[j] / densities[j]**2
                        viscosity = viscosities[i] + viscosities[j]

                        a_x = -mass * (term1 + term2 + viscosity) * grad_w * dx/r
                        a_y = -mass * (term1 + term2 + viscosity) * grad_w * dy/r
                        a_z = -mass * (term1 + term2 + viscosity) * grad_w * dz/r

                        acceleration_x[i] += a_x
                        acceleration_y[i] += a_y
                        acceleration_z[i] += a_z

        return np.vstack((acceleration_x, acceleration_y, acceleration_z)).T
    except Exception as e:
        print("Error en compute_acceleration_3d_partial:", e)
        traceback.print_exc()
        return None


def compute_artificial_viscosity_3d_partial_vectorized(positions, vx, vy, vz, densities, internal_energies, h_values, alpha=1, beta=2):
    """
    Vectorized version to compute acceleration for a subset of particles in 3D.

    :param positions: Positions of particles (Nx3 numpy array).
    :param densities: Densities of particles (numpy array).
    :param pressures: Pressures of particles (numpy array).
    :param mass: Mass of each particle (scalar).
    :param h_values: Smoothing lengths of particles (numpy array).
    :param viscosities: Viscosity values for particles (numpy array).
    :return: Acceleration vectors for each particle (Nx3 numpy array).
    """

    N = positions.shape[0]

    # Inicializa el array de viscosidad
    pi_c_i = np.zeros(N)

    # Calcula las diferencias en posición y velocidad
    dx = positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0]
    dy = positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1]
    dz = positions[:, np.newaxis, 2] - positions[np.newaxis, :, 2]
    dvx = vx[:, np.newaxis] - vx[np.newaxis, :]
    dvy = vy[:, np.newaxis] - vy[np.newaxis, :]
    dvz = vz[:, np.newaxis] - vz[np.newaxis, :]

    # Calcula la distancia entre partículas y evita la división por cero
    r_ij = np.sqrt(dx**2 + dy**2 + dz**2)
    r_ij[r_ij == 0] = np.inf

    # Calcula los términos intermedios
    c_ij = 0.5 * (np.sqrt(internal_energies[:, np.newaxis]) + np.sqrt(internal_energies[np.newaxis, :]))
    h_ij = 0.5 * (h_values[:, np.newaxis] + h_values[np.newaxis, :])
    condition = dvx*dx + dvy*dy + dvz*dz
    mu_ij = h_ij * condition / (r_ij**2 + 1e-5 * h_ij**2)
    rho_ij = 0.5 * (densities[:, np.newaxis] + densities[np.newaxis, :])

    # Calcula la viscosidad artificial para cada par de partículas
    pi_c_ij = np.where(condition < 0, -alpha * c_ij * mu_ij / rho_ij + beta * mu_ij**2, 0)

    # Suma los efectos de todas las demás partículas en cada partícula
    pi_c_i = np.sum(pi_c_ij, axis=1)

    return pi_c_i



def compute_acceleration_3d_partial_vectorized(positions, densities, pressures, mass, h_values, viscosities):
    """
    Computes accelerations for a subset of particles in a 3D space using a vectorized approach.

    This function calculates gravitational and hydrodynamic accelerations based on the Smoothed Particle Hydrodynamics (SPH) methodology. It uses a Gaussian kernel for smoothing and takes into account the artificial viscosity to simulate viscous effects.

    Parameters:
    - positions (numpy array): The positions of the particles in 3D space. This should be an Nx3 array where N is the number of particles.
    - densities (numpy array): The densities of the particles. This should be a 1D array of length N.
    - pressures (numpy array): The pressures at the positions of the particles. This should be a 1D array of length N.
    - mass (float): The mass of each particle. In SPH, all particles typically have the same mass.
    - h_values (numpy array): The smoothing lengths for each particle, used in the kernel function. This should be a 1D array of length N.
    - viscosities (numpy array): The viscosity values for each particle, used in the artificial viscosity calculation. This should be a 1D array of length N.

    Returns:
    - acceleration (numpy array): The calculated accelerations for each particle. This will be an Nx3 array where each row corresponds to the acceleration vector of a particle.

    The function uses numpy's advanced broadcasting and vectorized operations to efficiently compute accelerations for all particles. It calculates pairwise interactions between particles and sums up their contributions to find the total acceleration on each particle.
    """
    N = positions.shape[0]

    # Inicializa los arrays de aceleración
    acceleration = np.zeros_like(positions)

    # Calcula todas las diferencias de posición de una vez
    dx = positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0]
    dy = positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1]
    dz = positions[:, np.newaxis, 2] - positions[np.newaxis, :, 2]

    # Calcula las distancias entre todas las partículas
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Evita la división por cero
    r[r == 0] = np.inf

    # Calcula el gradiente del kernel
    grad_w = gradient_gaussian_kernel_vectorized(r, 0.5 * (h_values[:, np.newaxis] + h_values[np.newaxis, :]))

    # Calcula los términos de presión y viscosidad
    term1 = pressures / densities**2
    term2 = pressures[np.newaxis, :] / densities[np.newaxis, :]**2
    viscosity = viscosities[:, np.newaxis] + viscosities[np.newaxis, :]

    # Calcula las aceleraciones
    a = -mass * (term1[:, np.newaxis] + term2 + viscosity) * grad_w

    acceleration_x = np.sum(a * dx/r, axis=1)
    acceleration_y = np.sum(a * dy/r, axis=1)
    acceleration_z = np.sum(a * dz/r, axis=1)

    acceleration[:, 0] = acceleration_x
    acceleration[:, 1] = acceleration_y
    acceleration[:, 2] = acceleration_z

    return acceleration
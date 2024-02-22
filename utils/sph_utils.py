import numpy as np

def gaussian_kernel(r, h):
    """
    Gaussian kernel function for Smoothed Particle Hydrodynamics (SPH).

    This kernel is used to weight the contribution of nearby particles based on their distance.
    
    :param r: Distance from the particle (scalar or numpy array).
    :param h: Smoothing length (scalar).
    :return: Weighting value based on the Gaussian function (scalar or numpy array).
    """
    norm = 1 / (np.pi**(3/2) * h**3)
    return norm * np.exp(-r**2 / h**2)


def gradient_gaussian_kernel_vectorized(r, h):
    """
    Calculates the gradient of the Gaussian kernel function for SPH at a given distance r and smoothing length h.

    This gradient is used to compute forces and other vector quantities in SPH simulations.

    :param r: Distance from the particle (scalar or numpy array).
    :param h: Smoothing length (scalar).
    :return: Gradient of the Gaussian kernel (scalar or numpy array).
    """
    prefactor = -3 * r / (h**3 * np.sqrt(np.pi))
    exponential_part = np.exp(-r**2 / h**2)

    # Mask to identify non-infinite r values
    mask = np.isfinite(r)
    
    # Compute the gradient where r is finite
    result = np.zeros_like(r)
    result[mask] = prefactor[mask] * exponential_part[mask]

    return result


def gradient_gaussian_kernel(r, h):
    """
    Calculates the gradient of the Gaussian kernel function for SPH at a given distance r and smoothing length h.
    This version handles both scalar and numpy array inputs for r.

    :param r: Distance from the particle (scalar or numpy array).
    :param h: Smoothing length (scalar).
    :return: Gradient of the Gaussian kernel (scalar or numpy array).
    """
    # Si r es un escalar, realiza el cálculo directamente
    if np.isscalar(r):
        if np.isfinite(r):
            prefactor = -3 * r / (h**3 * np.sqrt(np.pi))
            exponential_part = np.exp(-r**2 / h**2)
            return prefactor * exponential_part
        return 0.0

    # Si r es un array, inicializa un array de resultados y realiza el cálculo para cada elemento
    result = np.zeros_like(r)
    for i in range(len(r)):
        if np.isfinite(r[i]):
            prefactor = -3 * r[i] / (h**3 * np.sqrt(np.pi))
            exponential_part = np.exp(-r[i]**2 / h**2)
            result[i] = prefactor * exponential_part

    return result


def compute_smoothing_length_density_based(masses, densities, eta=1.2, dimension=3):
    
    """
    Computes the adaptive smoothing length in 3D based on local density for SPH simulations.

    The smoothing length determines the spatial scale over which particle properties are smoothed.
    It is a crucial parameter in SPH simulations as it controls the resolution and accuracy.

    :param masses: Masses of the particles (numpy array).
    :param densities: Densities of the particles (numpy array).
    :param eta: Parameter controlling the number of neighbors. A larger eta increases the smoothing length,
                leading to smoother results but potentially lower resolution. Typically, eta is chosen
                to ensure an adequate number of neighbors within the smoothing length for accurate density estimation.
    :param dimension: Dimension of the space (3 for 3D).
    :return: Numpy array with adaptive smoothing lengths for each particle.

    Note:
    The smoothing length is calculated using a simple volume-based approach, where the volume of each particle
    is considered and the nth root (based on the dimension of the space) is taken. This approach adapts the
    smoothing length to local density variations, ensuring that regions with higher particle densities have
    smaller smoothing lengths, leading to higher resolution where it is most needed.
    """
    masses = np.asarray(masses)
    densities = np.asarray(densities)
    volumes = np.zeros_like(masses)

    # Calculate the volume represented by each particle
    np.divide(masses, densities, out=volumes, where=densities!=0)

    # Calculate the adaptive smoothing length
    smoothing_lengths = eta * np.power(volumes, (1.0 / dimension))
    
    return smoothing_lengths

def compute_adaptive_smoothing_length_adaptative(positions, densities, min_neighbors=20, max_neighbors=100):
    try:
        num_particles = positions.shape[0]
        smoothing_lengths = np.zeros(num_particles)
        density_mean = np.mean(densities)

        for i in range(num_particles):

            # Calcular un factor de escala basado en la densidad
            density_scale = densities[i] / density_mean

            # Ajustar el número de vecinos en función de la densidad
            desired_neighbors = int(np.clip(min_neighbors + (max_neighbors - min_neighbors) * density_scale, min_neighbors, max_neighbors))

            # Estimar el smoothing length (esta es una estimación simplificada, se puede mejorar)
            h = estimate_smoothing_length(positions[i], positions, desired_neighbors)
            smoothing_lengths[i] = h

        return smoothing_lengths
    except Exception as e:
        print("Error in compute_adaptive_smoothing_length_adaptative: ", e)
        return None

def estimate_smoothing_length(particle_position, positions, desired_neighbors):
    try:
        # Calcula las distancias a todas las demás partículas
        distances = np.linalg.norm(positions - particle_position, axis=1)
        sorted_distances = np.sort(distances)

        # Retorna la distancia al vecino deseado como una estimación del smoothing length
        return sorted_distances[desired_neighbors]
    except Exception as e:
        print("Error in estimate_smoothing_length: ", e)
        return None
    
def compute_smoothing_length_neighbors_based(positions, num_neighbors=59):
    try:
        """
        Calcula el smoothing length para cada partícula basándose en la distancia a sus 
        vecinos más cercanos.

        :param positions: Array de posiciones de las partículas (array de dimensiones (N, 3) para 3D).
        :param num_neighbors: Número de vecinos más cercanos a considerar.
        :return: Array con el smoothing length calculado para cada partícula.
        """
        num_particles = positions.shape[0]
        h_values = np.zeros(num_particles)
        # Calcula la distancia entre todas las partículas
        for i in range(num_particles):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            sorted_distances = np.sort(distances)
            
            # Considera la distancia al 'num_neighbors'-ésimo vecino más cercano como h
            if num_neighbors < num_particles:
                h_values[i] = sorted_distances[num_neighbors]
            else:
                h_values[i] = sorted_distances[-1]
        
        return h_values
    
    except Exception as e:
        print("Error in compute_smoothing_length_neighbors: ", e)
        return None
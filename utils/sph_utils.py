import numpy as np

def gaussian_kernel(r, h):
    """Kernel gaussiano para SPH."""
    norm = 1 / (np.pi**(3/2) * h**3)
    return norm * np.exp(-r**2 / h**2)

# @timed
def gradient_gaussian_kernel(r, h):
    """Calcula el gradiente de la función kernel gaussiana para un dado r y h."""
    prefactor = -3 * r / (h**3 * np.sqrt(np.pi))
    exponential_part = np.exp(-r**2 / h**2)

    # Máscara para identificar los valores de r que no son infinitos
    mask = np.isfinite(r)
    
    # Calcula el gradiente solo donde r no es infinito
    result = np.zeros_like(r)
    result[mask] = prefactor[mask] * exponential_part[mask]

    return result

def compute_smoothing_length_3d(masses, densities, eta=1.2, dimension=3):
    """
    Calcula la longitud de suavizado adaptativo en 3D basada en la densidad local.
    
    :param masses: Masa de las partículas (array de numpy).
    :param densities: Densidad de las partículas (array de numpy).
    :param eta: Parámetro que controla el número de vecinos (escalar).
    :param dimension: Dimensión del espacio (3 para 3D).
    :return: Array de numpy con los smoothing lengths adaptativos para cada partícula.
    """
    
    # Asegurarse de que las masas y las densidades son arrays de numpy para operaciones vectorizadas
    masses = np.asarray(masses)
    densities = np.asarray(densities)
    volumes = np.zeros_like(masses)
    # Calcular el volumen que representa cada partícula
    np.divide(masses, densities, out=volumes, where=densities!=0)
    
    # Calcular la longitud de suavizado adaptativo
    smoothing_lengths = eta * np.power(volumes, ((1.0 / dimension)))
    
    return smoothing_lengths
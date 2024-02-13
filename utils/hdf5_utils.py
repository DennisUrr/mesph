import h5py
import os
import shutil
from math import pi
import numpy as np


def create_unique_directory(base_path, args):
    """
    Creates a unique directory with specified abbreviated arguments in its name. 
    If the base directory already exists, it creates a new one with an incremental index.
    :param base_path: The base path of the directory to be created.
    :param args: Arguments to be included in the directory name.
    :return: The path of the created directory.
    """

    # Mapeo de argumentos a sus abreviaturas
    arg_map = {
        'processors': 'p',
        'particles': 'n',
        'alpha': 'a',
        'beta': 'b',
        'eta': 'eta',
        'extrapolation': 'e',
        'total_files': 'tf',
        'dT_initial': 'dti',
        'dT_final': 'dtf',
        'mode': 'm',
        'smoothig_length_mode': 'hm',
        'vectorized_mode': 'vm',
        'dust_mode': 'dm'
    }

    # Función para crear una cadena con las abreviaturas de los argumentos
    def create_arg_string(args):
        arg_items = vars(args).items()
        return '_'.join(f"{arg_map.get(key, key)}{value}" for key, value in arg_items if value is not None and key in arg_map)

    # Incluye los argumentos en el nombre del directorio
    arg_str = create_arg_string(args)
    base_path_with_args = f"{base_path}_{arg_str}"
    
    counter = 0
    new_path = base_path_with_args
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path_with_args}_{counter}"
    
    os.makedirs(new_path)
    return new_path



def create_snapshot_file(dT, file_idx, Ntot, positions_3d, velocities, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir, start_idx, end_idx):
    """
    Creates a snapshot file for a specific time and file index with particle data in HDF5 format.

    :param dT: Time step of the snapshot.
    :param file_idx: File index for the current snapshot part.
    :param Ntot: Total number of particles.
    :param positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities: Particle properties arrays.
    :param base_filename: The base name for the snapshot files.
    :param total_files: Total number of files per snapshot.
    :param unique_dir: The directory where the snapshot file will be saved.

    The function creates a snapshot file named with the time step and file index, containing all the provided particle properties.
    """
    # Assign particle IDs
    ids = np.arange(start_idx, end_idx, dtype=np.int32)
    # Define the full filename with the path to the unique directory
    filename = os.path.join(unique_dir, f'{base_filename}_{int(dT):03d}.{file_idx}.hdf5')

    with h5py.File(filename, 'w') as f:
        # Create the Header group and set attributes
        header = f.create_group("/Header")
        header.attrs['NumPart_ThisFile'] = [Ntot, 0, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [Ntot * total_files, 0, 0, 0, 0, 0]  # Ntot * total_files is the total number of particles across all files
        header.attrs['NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
        header.attrs['MassTable'] = [0, 0, 0, 0, 0, 0]
        header.attrs['Time'] = int(dT)*pi
        header.attrs['Redshift'] = 0
        header.attrs["NumFilesPerSnapshot"] = total_files
        header.attrs["Dimension"] = 3

        # Create the PartType0 group and add datasets
        pt0 = f.create_group("/PartType0") # GAS
        pt0.create_dataset("Coordinates", data=positions_3d)
        pt0.create_dataset("Velocities", data=velocities)
        pt0.create_dataset("ParticleIDs", data=ids)
        pt0.create_dataset("Masses", data=masses)
        pt0.create_dataset("InternalEnergy", data=particle_energies)
        pt0.create_dataset("Density", data=densities)
        pt0.create_dataset("SmoothingLength", data=h_values)
        pt0.create_dataset("Acceleration", data=accelerations)
        pt0.create_dataset("Pressure", data=pressures)
        pt0.create_dataset("Viscosity", data=viscosities)

    #print("==============================================================\n", "========= HDF5 File "+ str(dT) +"."+ str(file_idx) +" Created. =========\n", "==============================================================")
    #print(dT, '.' ,file_idx, " Created.")

def copy_files_to_directory(source_files, destination_directory):
    """
    Copies files to a destination directory.
    
    :param source_files: List of file paths to copy.
    :param destination_directory: Path of the destination directory.
    """
    for file in source_files:
        # Ensure the source file exists before attempting to copy
        if os.path.isfile(file):
            shutil.copy(file, destination_directory)
        else:
            print(f"The file {file} does not exist and cannot be copied.")


def write_to_file(dT, file_idx, combined_results, unique_dir, total_files, base_filename="snapshot_3d"):
    """
    Writes the combined results of particle properties processed by a CPU into a single HDF5 file.

    :param dT: Time step of the snapshot.
    :param file_idx: File index for the current snapshot part.
    :param combined_results: A tuple containing combined arrays of all particle properties.
    :param base_filename: The base name for the snapshot files.
    :param unique_dir: The directory where the snapshot file will be saved.
    """
    # Desempaquetar los resultados combinados
    positions, velocities, masses, energies, densities, h_values, accelerations, pressures, viscosities = combined_results

    
    # Calcular el número total de partículas
    Ntot = positions.shape[0]
    # Calcular los IDs de las partículas
    ids = np.arange(Ntot, dtype=np.int32)

    # Definir el nombre completo del archivo con la ruta al directorio único
    filename = os.path.join(unique_dir, f'{base_filename}_{int(dT):03d}.{file_idx}.hdf5')

    with h5py.File(filename, 'w') as f:
        # Create the Header group and set attributes
        header = f.create_group("/Header")
        header.attrs['NumPart_ThisFile'] = [Ntot, 0, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [Ntot * total_files, 0, 0, 0, 0, 0]  # Ntot * total_files is the total number of particles across all files
        header.attrs['NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
        header.attrs['MassTable'] = [0, 0, 0, 0, 0, 0]
        header.attrs['Time'] = int(dT)*pi
        header.attrs['Redshift'] = 0
        header.attrs["NumFilesPerSnapshot"] = total_files
        header.attrs["Dimension"] = 3

        # Create the PartType0 group and add datasets
        pt0 = f.create_group("/PartType0") # GAS
        pt0.create_dataset("Coordinates", data=positions)
        pt0.create_dataset("Velocities", data=velocities)
        pt0.create_dataset("ParticleIDs", data=ids)
        pt0.create_dataset("Masses", data=masses)
        pt0.create_dataset("InternalEnergy", data=energies)
        pt0.create_dataset("Density", data=densities)
        pt0.create_dataset("SmoothingLength", data=h_values)
        pt0.create_dataset("Acceleration", data=accelerations)
        pt0.create_dataset("Pressure", data=pressures)
        pt0.create_dataset("Viscosity", data=viscosities)

def write_to_file_dust(dT, file_idx, combined_results, unique_dir, total_files, base_filename="snapshot_3d"):
    """
    Writes the combined results of particle properties processed by a CPU into a single HDF5 file.

    :param dT: Time step of the snapshot.
    :param file_idx: File index for the current snapshot part.
    :param combined_results: A tuple containing combined arrays of all particle properties.
    :param base_filename: The base name for the snapshot files.
    :param unique_dir: The directory where the snapshot file will be saved.
    """
    # Desempaquetar los resultados combinados
    positions, velocities, masses, densities, h_values = combined_results

    
    # Calcular el número total de partículas
    Ntot = positions.shape[0]
    # Calcular los IDs de las partículas
    ids = np.arange(Ntot, dtype=np.int32)

    # Definir el nombre completo del archivo con la ruta al directorio único
    filename = os.path.join(unique_dir, f'{base_filename}_dust_{int(dT):03d}.{file_idx}.hdf5')

    with h5py.File(filename, 'w') as f:
        # Create the Header group and set attributes
        header = f.create_group("/Header")
        header.attrs['NumPart_ThisFile'] = [Ntot, 0, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [Ntot * total_files, 0, 0, 0, 0, 0]  # Ntot * total_files is the total number of particles across all files
        header.attrs['NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
        header.attrs['MassTable'] = [0, 0, 0, 0, 0, 0]
        header.attrs['Time'] = int(dT)*pi
        header.attrs['Redshift'] = 0
        header.attrs["NumFilesPerSnapshot"] = total_files
        header.attrs["Dimension"] = 3

        # Create the PartType0 group and add datasets
        pt0 = f.create_group("/PartType0") # Dust
        pt0.create_dataset("Coordinates", data=positions)
        pt0.create_dataset("Velocities", data=velocities)
        pt0.create_dataset("ParticleIDs", data=ids)
        pt0.create_dataset("Masses", data=masses)
        pt0.create_dataset("Density", data=densities)
        pt0.create_dataset("SmoothingLength", data=h_values)
        
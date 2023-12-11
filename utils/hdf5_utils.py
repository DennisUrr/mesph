import h5py
import os
import shutil
from math import pi

def create_unique_directory(base_path):
    """
    Crea un directorio único. Si el directorio base ya existe, crea uno nuevo con un índice incremental.
    :param base_path: La ruta base del directorio a crear.
    :return: La ruta del directorio creado.
    """
    counter = 0
    new_path = base_path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    
    os.makedirs(new_path)
    return new_path

def create_snapshot_file(dT, file_idx, Ntot, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir):

    # Define el nombre de archivo completo con la ruta al directorio único
    filename = os.path.join(unique_dir, f'{base_filename}_{int(dT):03d}.{file_idx}.hdf5')

    with h5py.File(filename, 'w') as f:
        # Crear el grupo Header y establecer los atributos
        header = f.create_group("/Header")
        header.attrs['NumPart_ThisFile'] = [Ntot, 0, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [Ntot * total_files, 0, 0, 0, 0, 0]  # Ntot * total_files es el total de partículas en todos los archivos
        header.attrs['NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
        header.attrs['MassTable'] = [0, 0, 0, 0, 0, 0]
        header.attrs['Time'] = int(dT)*pi
        header.attrs['Redshift'] = 0
        header.attrs["NumFilesPerSnapshot"] = total_files
        header.attrs["Dimension"] = 3

        # Crear el grupo PartType0 y agregar los datasets
        pt0 = f.create_group("/PartType0")
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



def copy_files_to_directory(source_files, destination_directory):
    """
    Copia archivos a un directorio destino.
    
    :param source_files: Lista de rutas de archivos a copiar.
    :param destination_directory: Ruta del directorio destino.
    """
    for file in source_files:
        # Asegúrate de que el archivo de origen existe antes de intentar copiarlo
        if os.path.isfile(file):
            shutil.copy(file, destination_directory)
        else:
            print(f"El archivo {file} no existe y no se puede copiar.")
import h5py
import os
import shutil
from math import pi

def create_unique_directory(base_path):
    """
    Creates a unique directory. If the base directory already exists, it creates a new one with an incremental index.
    :param base_path: The base path of the directory to be created.
    :return: The path of the created directory.
    """
    counter = 0
    new_path = base_path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    
    os.makedirs(new_path)
    return new_path

def create_snapshot_file(dT, file_idx, Ntot, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir):
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
        
    #print("==============================================================\n", "========= HDF5 File "+ str(dT) +"."+ str(file_idx) +" Created. =========\n", "==============================================================")


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

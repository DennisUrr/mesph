import time
import numpy as np
import os
from multiprocessing import cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.parameters import read_parameters
from utils.conversions import to_cartesian_3d, velocities_to_cartesian_3d
from utils.sampling import sample_particles_3d_partial, assign_particle_velocities_from_grid_3d_partial, assign_particle_densities_from_grid_3d_partial, assign_particle_internal_energies_from_grid_3d_partial, sample_particles_3d_interpolated
from utils.physics import compute_pressure, compute_acceleration_3d_partial, compute_artificial_viscosity_3d_partial, compute_particle_mass_3d
from utils.sph_utils import compute_smoothing_length_3d
from utils.hdf5_utils import create_snapshot_file, create_unique_directory, copy_files_to_directory
from run_splash import run_splash
import argparse
from tqdm import tqdm

def sample_and_convert_particles(task):
    """
    Samples and converts a subset of particles from spherical to Cartesian coordinates.
    :param task: A tuple containing the necessary parameters for particle sampling.
    :return: Positions in Cartesian coordinates, and lists of r, phi, and theta.
    """
    (file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, total_cpus, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir) = task

    rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
    x, y, z = to_cartesian_3d(rlist, philist, thetalist)
    positions_3d = np.column_stack((x, y, z))
    return positions_3d, rlist, philist, thetalist


def calculate_velocities_densities_energies(rho, vphi, vr, vtheta, rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx):
    """
    Calculates velocities, densities, and internal energies for a subset of particles.
    :param rho, vphi, vr, vtheta, rlist, philist, thetalist, u, rmed, phimed, thetamed: Parameters for calculating properties.
    :return: Velocities, densities, and internal energies of particles.
    """
    vrlist, vphilist, vthetalist = assign_particle_velocities_from_grid_3d_partial(vphi, vr, vtheta, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
    vx, vy, vz = velocities_to_cartesian_3d(vrlist, vphilist, vthetalist, rlist, philist, thetalist)
    velocities = np.column_stack((vx, vy, vz))
    densities = assign_particle_densities_from_grid_3d_partial(rho, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
    particle_energies = assign_particle_internal_energies_from_grid_3d_partial(rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx)
    return velocities, densities, particle_energies


def calculate_masses_pressures_smoothing_lengths(nr, ntheta, rho, ASPECTRATIO, params, Ntot, Ntot_per_file, positions_3d, velocities, densities, particle_energies, gamma, alpha, beta):
    """
    Calculates particle masses, pressures, smoothing lengths, viscosities, and accelerations.
    :param nr, ntheta, rho, ASPECTRATIO, params, Ntot, Ntot_per_file, positions_3d, velocities, densities, particle_energies, gamma, alpha, beta: Parameters for calculations.
    :return: Masses, pressures, smoothing lengths, viscosities, and accelerations of particles.
    """
    particle_mass = compute_particle_mass_3d(nr, ntheta, rho, ASPECTRATIO, params, Ntot)
    masses = np.full(Ntot_per_file, particle_mass, dtype=np.float32)
    pressures = compute_pressure(densities, particle_energies, gamma)
    h_values = compute_smoothing_length_3d(masses, densities)
    viscosities = compute_artificial_viscosity_3d_partial(positions_3d, velocities[:, 0], velocities[:, 1], velocities[:, 2], densities, particle_energies, h_values, alpha, beta)
    accelerations = compute_acceleration_3d_partial(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
    return masses, pressures, h_values, viscosities, accelerations

def create_hdf5_file(dT, file_idx, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, unique_dir, base_filename = 'snapshot_3d'):
    create_snapshot_file(dT, file_idx, Ntot_per_file, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_cpus, unique_dir)


def process_file(file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, total_cpus, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir, base_filename = 'snapshot_3d'): 
    
    # Sampling the particles for the actual subset
    rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
    #rlist, philist, thetalist = sample_particles_3d_interpolated(rho, rmed, phimed, thetamed, start_idx, end_idx)
    
    # Convert spherical coordinates to Cartesian coordinates
    x, y, z = to_cartesian_3d(rlist, philist, thetalist)
    positions_3d = np.column_stack((x, y, z))

    
    # Convert velocity components from spherical to Cartesian coordinates
    vrlist, vphilist, vthetalist = assign_particle_velocities_from_grid_3d_partial(vphi, vr, vtheta, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
    vx, vy, vz = velocities_to_cartesian_3d(vrlist, vphilist, vthetalist, rlist, philist, thetalist)
    velocities = np.column_stack((vx, vy, vz))
    
    # Assign densities and internal energies to particles
    densities = assign_particle_densities_from_grid_3d_partial(rho, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
    particle_energies = assign_particle_internal_energies_from_grid_3d_partial(rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx)


    # Compute particle masses
    particle_mass = compute_particle_mass_3d(nr, ntheta, rho, ASPECTRATIO, params, Ntot)
    masses = np.full(Ntot_per_file, particle_mass, dtype=np.float32)

    # Compute particle pressures     
    pressures = compute_pressure(densities, particle_energies, gamma)
    
    # Compute smoothing lengths
    h_values = compute_smoothing_length_3d(masses, densities)
    
    # Compute viscosities
    viscosities = compute_artificial_viscosity_3d_partial(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)

    # Compute accelerations    
    accelerations = compute_acceleration_3d_partial(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
    
    # Assign particle IDs
    ids = np.arange(start_idx, end_idx, dtype=np.int32)
    
    # Create snapshot file
    create_snapshot_file(dT, file_idx, Ntot_per_file, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir)


def main_parallel_1(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, total_files):
    dT=str(0)

    global particle_mass
    global params, gamma, ASPECTRATIO, Ntot_per_file, phi, r, theta, phimed, rmed, thetamed, nphi, nr, ntheta

    unique_dir = create_unique_directory(output_dir)
    # print(f"Unique directory created: {unique_dir}")

    params = read_parameters( path_outputs_fargo + "/variables.par")
    gamma = float(params['GAMMA'])
    ASPECTRATIO = float(params['ASPECTRATIO'])
    Ntot_per_file = Ntot // total_files  # number of particles per file
    Ntot = Ntot_per_file * total_files  # total number of particles

    # load the domain of the FARGO3D simulation
    phi = np.loadtxt( path_outputs_fargo + "/domain_x.dat")
    r = np.loadtxt( path_outputs_fargo + "domain_y.dat")[3:-4]
    theta = np.loadtxt( path_outputs_fargo + "domain_z.dat")[3:-3]

    # compute the midpoints for the angle and radius
    phimed = 0.5*(phi[1:]+phi[:-1])
    rmed   = 0.5*(r[1:]+r[:-1])
    thetamed = 0.5*(theta[1:]+theta[:-1])

    # lens fot phi, r and theta
    nphi = len(phimed)
    nr   = len(rmed)
    ntheta = len(theta)

    tasks = []
    for dT in range(total_timesteps):
        dT = str(dT)

        # load the gas density
        rho = np.fromfile( path_outputs_fargo + '/gasdens' + dT + '.dat').reshape(len(theta)-1,len(r),len(phi)-1)#volume density

        # load the gas velocities
        vphi = np.fromfile( path_outputs_fargo + '/gasvx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vr = np.fromfile( path_outputs_fargo + '/gasvy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vtheta = np.fromfile( path_outputs_fargo + '/gasvz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        # load the gas internal energy
        u = np.fromfile( path_outputs_fargo + '/gasenergy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        # print("FARGO3D files loaded")
        for file_idx in range(total_files):
            start_idx = file_idx * Ntot_per_file
            end_idx = start_idx + Ntot_per_file if file_idx != total_files - 1 else Ntot
            # Add a task for each file
            tasks.append((file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, total_files, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir))

    # create a process pool and execute process_file in parallel for all tasks
    with Pool(total_cpus) as pool:
        #pool.starmap(process_file, tasks)
        results = pool.map(sample_and_convert_particles, tasks)


    for result, task in zip(results, tasks):
        (file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, total_cpus, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir) = task
        positions_3d, rlist, philist, thetalist = result
        velocities, densities, particle_energies = calculate_velocities_densities_energies(rho, vphi, vr, vtheta, rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx)
        masses, pressures, h_values, viscosities, accelerations = calculate_masses_pressures_smoothing_lengths(nr, ntheta, rho, ASPECTRATIO, params, Ntot, Ntot_per_file, positions_3d, velocities, densities, particle_energies, gamma, alpha, beta)
        ids = np.arange(0, end_idx, dtype=np.int32)
        create_hdf5_file(dT, file_idx, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, unique_dir, base_filename='snapshot_3d')

    source_files = ['outputs/splash.defaults', 'outputs/splash.limits']
    destination_directory = unique_dir

    if not os.path.exists(destination_directory):
        print(f"El directorio {destination_directory} no existe.")

    copy_files_to_directory(source_files, destination_directory)
    run_splash(total_timesteps, total_files, unique_dir)


def main_parallel_2(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, total_files):
    dT=str(0)

    global particle_mass
    global params, gamma, ASPECTRATIO, Ntot_per_file, phi, r, theta, phimed, rmed, thetamed, nphi, nr, ntheta

    unique_dir = create_unique_directory(output_dir)
    print(f"Unique directory created: {unique_dir}")

    params = read_parameters( path_outputs_fargo + "/variables.par")
    gamma = float(params['GAMMA'])
    ASPECTRATIO = float(params['ASPECTRATIO'])
    Ntot_per_file = Ntot // total_files  # number of particles per file
    Ntot = Ntot_per_file * total_files  # total number of particles

    # load the domain of the FARGO3D simulation
    phi = np.loadtxt( path_outputs_fargo + "/domain_x.dat")
    r = np.loadtxt( path_outputs_fargo + "domain_y.dat")[3:-4]
    theta = np.loadtxt( path_outputs_fargo + "domain_z.dat")[3:-3]

    # compute the midpoints for the angle and radius
    phimed = 0.5*(phi[1:]+phi[:-1])
    rmed   = 0.5*(r[1:]+r[:-1])
    thetamed = 0.5*(theta[1:]+theta[:-1])

    # lens fot phi, r and theta
    nphi = len(phimed)
    nr   = len(rmed)
    ntheta = len(theta)

    tasks = []
    for dT in range(total_timesteps):
        dT = str(dT)

        # load the gas density
        rho = np.fromfile( path_outputs_fargo + '/gasdens' + dT + '.dat').reshape(len(theta)-1,len(r),len(phi)-1)#volume density

        # load the gas velocities
        vphi = np.fromfile( path_outputs_fargo + '/gasvx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vr = np.fromfile( path_outputs_fargo + '/gasvy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vtheta = np.fromfile( path_outputs_fargo + '/gasvz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        # load the gas internal energy
        u = np.fromfile( path_outputs_fargo + '/gasenergy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        # print("FARGO3D files loaded")
        for file_idx in range(total_files):
            start_idx = file_idx * Ntot_per_file
            end_idx = start_idx + Ntot_per_file if file_idx != total_files - 1 else Ntot
            # Add a task for each file
            tasks.append((file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, total_cpus, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir))

    # # create a process pool and execute process_file in parallel for all tasks
    # with Pool(total_cpus) as pool:
    #     #pool.starmap(process_file, tasks)
    #     pool.starmap(process_file, tasks)
    with ProcessPoolExecutor(max_workers=total_cpus) as executor:
        futures = [executor.submit(process_file, *task) for task in tasks]
        with tqdm(total=len(tasks), desc='Processing files') as progress_bar:
            for future in as_completed(futures):
                progress_bar.update(1)
    

    source_files = ['outputs/splash.defaults', 'outputs/splash.limits']
    destination_directory = unique_dir

    if not os.path.exists(destination_directory):
        print(f"El directorio {destination_directory} no existe.")

    copy_files_to_directory(source_files, destination_directory)
    run_splash(total_timesteps, total_files, unique_dir)


if __name__ == '__main__':
    init_time = time.time()
    
     # Create an argument parser
    parser = argparse.ArgumentParser(description='Parallel particle processing for astrophysical simulations.')
    
    # Number of processors to be used for parallel processing
    parser.add_argument('-p', '--processors', type=int, default=2, help='Number of processors to use for parallel processing.')
    
    # Output directory for the generated HDF5 files
    parser.add_argument('-o', '--output', type=str, default='outputs/snapshot', help='Output directory for the generated HDF5 files.')
    
    # Directory containing FARGO3D output files
    parser.add_argument('-of', '--output_fargo', type=str, default='../public/outputs/p3disof/', help='Directory containing FARGO3D output files.')
    
    # Number of time steps to process
    parser.add_argument('-t', '--times', type=int, default=1, help='Number of time steps to process.')

    # Number of particles to transform
    parser.add_argument('-n', '--particles', type=int, default=10000, help='Number of particles to transform.')

    # Alpha parameter for artificial viscosity
    parser.add_argument('-a', '--alpha', type=float, default=0.6, help='Alpha parameter for artificial viscosity.')
    
    # Beta parameter for artificial viscosity
    parser.add_argument('-b', '--beta', type=float, default=1, help='Beta parameter for artificial viscosity.')
    
    parser.add_argument('-tf', '--total_files', type=float, default=2, help='Number of files to create 0->(-tf)-1.')

    # Type of parallelization
    parser.add_argument('-m', '--mode', type=int, default=1, help='Type of parallelization.')
    args = parser.parse_args()
    total_cpus = args.processors
    output_dir = args.output
    path_outputs_fargo = args.output_fargo
    total_timesteps = args.times
    Ntot = args.particles
    alpha = args.alpha
    beta = args.beta
    total_files = int(args.total_files)
    mode = args.mode
    
    # print(f"Total CPUs: {total_cpus}")
    # print(f"total_files: {total_files}")

    if mode==1:
        main_parallel_1(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, total_files)
    elif mode==2:
        main_parallel_2(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, total_files)
    else:
        print("Invalid mode. Please choose 1 or 2.")

    print(f"Tiempo total: {round(time.time() - init_time, 5)} segundos.")
    



    
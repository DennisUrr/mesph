import time
import numpy as np
import os
from multiprocessing import cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.parameters import read_parameters
from utils.conversions import spherical_to_cartesian, velocities_to_cartesian_3d, to_cartesian
from utils.sampling import sample_particles_3d_partial, assign_particle_velocities_from_grid_3d_partial, assign_particle_densities_from_grid_3d_partial, assign_particle_internal_energies_from_grid_3d_partial, sample_from_density_grid
from utils.physics import compute_pressure, compute_acceleration_3d_partial, compute_artificial_viscosity_3d_partial, compute_particle_mass_3d, compute_artificial_viscosity_3d_partial_vectorized, compute_acceleration_3d_partial_vectorized
from utils.sph_utils import compute_smoothing_length_3d
from utils.hdf5_utils import create_snapshot_file, create_unique_directory, copy_files_to_directory
from run_splash import run_splash
import argparse
from tqdm import tqdm

def process_file(file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, extrapolation_mode, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir, total_files, base_filename = 'snapshot_3d'): 
    
    try:
        # Sampling the particles for the actual subset
        if extrapolation_mode == 0:
            rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            x, y, z = spherical_to_cartesian(rlist, philist, thetalist)
        elif extrapolation_mode == 1:
            rlist, philist, thetalist = sample_particles_by_importance(rho, r, phi, theta, start_idx, end_idx)
            x, y, z = spherical_to_cartesian(rlist, philist, thetalist)
        else:
            rlist, philist, thetalist = sample_from_density_grid(rho, r, phi, theta, start_idx, end_idx)
            x, y, z = to_cartesian(rlist, philist, thetalist)
        
        #print("rlist.shape: ", rlist.shape)
        # Convert spherical coordinates to Cartesian coordinates
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
        viscosities = compute_artificial_viscosity_3d_partial_vectorized(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)

        # Compute accelerations    
        accelerations = compute_acceleration_3d_partial_vectorized(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
        
        # Assign particle IDs
        ids = np.arange(start_idx, end_idx, dtype=np.int32)
        
        # Create snapshot file
        create_snapshot_file(dT, file_idx, Ntot_per_file, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir)

    except Exception as e:
        print(f"Error processing file {file_idx} at time step {dT}: {e}")
        raise e

def main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files):
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
            tasks.append((file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, extrapolation_mode, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir, total_files))

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
    run_splash(total_timesteps, unique_dir)


if __name__ == '__main__':
    init_time = time.time()
    
     # Create an argument parser
    parser = argparse.ArgumentParser(description='Parallel particle processing for astrophysical simulations.')
    
    # Number of processors to be used for parallel processing
    parser.add_argument('-p', '--processors', type=int, default=2, help='Number of processors to use for parallel processing.')
    
    # Output directory for the generated HDF5 files
    parser.add_argument('-o', '--output', type=str, default='outputs/snapshot', help='Output directory for the generated HDF5 files.')
    
    # Directory containing FARGO3D output files
    parser.add_argument('-of', '--output_fargo', type=str, default='../FARGO3D/public/outputs/p3disof/', help='Directory containing FARGO3D output files.')
    
    # Number of time steps to process
    parser.add_argument('-t', '--times', type=int, default=1, help='Number of time steps to process.')

    # Number of particles to transform
    parser.add_argument('-n', '--particles', type=int, default=10000, help='Number of particles to transform.')

    # Alpha parameter for artificial viscosity
    parser.add_argument('-a', '--alpha', type=float, default=0.6, help='Alpha parameter for artificial viscosity.')
    
    # Beta parameter for artificial viscosity
    parser.add_argument('-b', '--beta', type=float, default=1, help='Beta parameter for artificial viscosity.')
    
    parser.add_argument('-tf', '--total_files', type=int, default=2, help='Number of files to create 0->(-tf)-1.')

    # Extrapolation mode
    parser.add_argument('-e', '--extrapolation', type=int, default=0, help='Extrapolation method: 0 -> probabilistic method, 1 -> trilineal cilindric interpolation, 2 -> method.')

    # Type of parallelization
    args = parser.parse_args()
    total_cpus = args.processors
    output_dir = args.output
    path_outputs_fargo = args.output_fargo
    total_timesteps = args.times
    Ntot = args.particles
    alpha = args.alpha
    beta = args.beta
    extrapolation_mode = args.extrapolation
    total_files = int(args.total_files)
    
    # print(f"Total CPUs: {total_cpus}")
    # print(f"total_files: {total_files}")
    main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files)
    
    print(f"Tiempo total: {round(time.time() - init_time, 5)} segundos.")
    



    
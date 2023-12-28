import time
import numpy as np
import os
from multiprocessing import cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.parameters import read_parameters
from processing import process_file
from utils.hdf5_utils import create_unique_directory, copy_files_to_directory
from run_splash import run_splash
import argparse
from tqdm import tqdm

def main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files, dT_initial=None, dT_final=None):
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
    if dT_initial == None and dT_final == None:
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
    
    elif dT_final == None:
        
        for dT in range(dT_initial, total_timesteps):
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
        total_timesteps = total_timesteps - dT_initial
    else:
        
        for dT in range(dT_initial, dT_final+1):
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
        total_timesteps = dT_final - dT_initial
        
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

    parser.add_argument('-dti', '--dT_initial', type=int, default=None, help='Initial time step to process.')
    
    parser.add_argument('-dtf', '--dT_final', type=int, default=None, help='Final time step to process.')

    parser.add_argument('-m', '--mode', type=int, default=0, help='Mode: 0 = 0 -> t, 1 = t_initial -> t, 2 = t_initial -> t_final.')


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
    dT_initial = int(args.dT_initial)
    dT_final = int(args.dT_final)
    mode = int(args.mode)

    # print(f"Total CPUs: {total_cpus}")
    # print(f"total_files: {total_files}")
    if mode == 0:
        dT_initial = None
        dT_final = None
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files)
        
    elif mode == 1:
        dT_final = None
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files, dT_initial)
        
    else:
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, extrapolation_mode, total_files, dT_initial, dT_final)
        
    
    print(f"Tiempo total: {round(time.time() - init_time, 5)} segundos.")


        
import time
import numpy as np
import os
from multiprocessing import cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.parameters import read_parameters
from processing import process_file, combine_and_write_results
from utils.hdf5_utils import create_unique_directory, copy_files_to_directory, write_to_file
from run_splash import run_splash, generate_shell_script_measures_errors
import argparse
from tqdm import tqdm


def get_dT_range(mode, total_timesteps, dT_initial, dT_final):
    if mode == 0:
        return range(0, total_timesteps)
    elif mode == 1:
        return range(dT_initial, dT_initial + total_timesteps)
    else:
        return range(dT_initial, dT_final)
    
def create_tasks(adjusted_dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, 
                 rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, 
                 nr, ntheta, total_files, h_mode, vectorized_mode, dust_mode, FRAME="F", OMEGAFRAME=0,
                 rho_dust=None, vphi_dust=None, vr_dust=None, vtheta_dust=None):
    tasks = []
    for file_idx in range(total_files):
        subset_size = Ntot // (total_files * total_cpus)
        Ntot_adjusted = subset_size * total_cpus * total_files
        for proc_idx in range(total_cpus):
            start_idx = proc_idx * subset_size + (file_idx * subset_size * total_cpus)
            end_idx = start_idx + subset_size
            # Crea la tarea dependiendo del modo
            if dust_mode == 0:  # Solo gas
                tasks.append((file_idx, adjusted_dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, 
                              Ntot_adjusted, subset_size, rho, phi, theta, r, phimed, rmed, thetamed, 
                              vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, h_mode, vectorized_mode, dust_mode, FRAME, OMEGAFRAME))
            else:  # Gas y polvo
                tasks.append((file_idx, adjusted_dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, 
                              Ntot_adjusted, subset_size, rho, phi, theta, r, phimed, rmed, thetamed, 
                              vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, h_mode, vectorized_mode, dust_mode, FRAME, OMEGAFRAME,
                              rho_dust, vphi_dust, vr_dust, vtheta_dust))
    return tasks


def main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, eta, extrapolation_mode, total_files, h_mode, vectorized_mode, mode, dust_mode, args, dT_initial=None, dT_final=None):
    
    dT=str(0)

    global particle_mass
    global params, gamma, ASPECTRATIO, Ntot_per_file, phi, r, theta, phimed, rmed, thetamed, nphi, nr, ntheta

    unique_dir = create_unique_directory(output_dir, args)
    print(f"Unique directory created: {unique_dir}")

    params = read_parameters( path_outputs_fargo + "/variables.par")
    gamma = float(params['GAMMA'])
    ASPECTRATIO = float(params['ASPECTRATIO'])
    FRAME = str(params['FRAME'])
    OMEGAFRAME = float(params['OMEGAFRAME']) if FRAME == "G" else 0


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

    dT_range = get_dT_range(mode, total_timesteps, dT_initial, dT_final)
    
    total_estimated_tasks = len(dT_range) * total_files

    with tqdm(total=total_estimated_tasks, desc="Overall Progress") as progress_bar:
        for idx, dT in enumerate(dT_range):
            
            dT = str(dT)

            # load the gas density
            rho = np.fromfile( path_outputs_fargo + '/gasdens' + dT + '.dat').reshape(len(theta)-1,len(r),len(phi)-1)#volume density
            
            # load the gas velocities
            vphi = np.fromfile( path_outputs_fargo + '/gasvx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
            vr = np.fromfile( path_outputs_fargo + '/gasvy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
            vtheta = np.fromfile( path_outputs_fargo + '/gasvz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
            
            # load the gas internal energy
            u = np.fromfile( path_outputs_fargo + '/gasenergy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
            
            # Ajusta dT según el modo
            adjusted_dT = str(int(dT) - dT_initial) if mode != 0 else str(dT)

            # Crea las tareas para cada archivo y cada subconjunto de partículas
            output_dT = idx
            
            if dust_mode == 0:
                tasks = create_tasks(adjusted_dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, total_files, h_mode, vectorized_mode, dust_mode)
            
            elif dust_mode == 1:
                # load the dust density
                rho_dust = np.fromfile( path_outputs_fargo + '/dust1dens' + dT + '.dat').reshape(len(theta)-1,len(r),len(phi)-1)
                
                # load the dust velocities
                vphi_dust = np.fromfile( path_outputs_fargo + '/dust1vx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
                vr_dust = np.fromfile( path_outputs_fargo + '/dust1vy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
                vtheta_dust = np.fromfile( path_outputs_fargo + '/dust1vz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
                tasks = create_tasks(adjusted_dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, total_files, h_mode, vectorized_mode, dust_mode, FRAME, OMEGAFRAME, rho_dust, vphi_dust, vr_dust, vtheta_dust)
            with ProcessPoolExecutor(max_workers=total_cpus) as executor:
                futures = [executor.submit(process_file, *task) for task in tasks]
                all_results = [future.result() for future in as_completed(futures)]

            combine_and_write_results(all_results, unique_dir, output_dT, total_files, dust_mode)
            progress_bar.update(total_files)
    progress_bar.close()

    source_files = ['outputs/splash.defaults', 'outputs/splash.limits']    
    destination_directory = unique_dir


    if not os.path.exists(destination_directory):
        print(f"El directorio {destination_directory} no existe.")

    copy_files_to_directory(source_files, destination_directory)
    run_splash(total_timesteps, unique_dir, dust_mode)
    generate_shell_script_measures_errors(unique_dir , int(params['NX']), int(params['NY']), int(params['NZ']))

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
    
    # eta parameter for smoothing length
    parser.add_argument('-eta', '--eta', type=float, default=1.2, help='Eta parameter for smoothing length.')

    # Number of files to create
    parser.add_argument('-tf', '--total_files', type=int, default=2, help='Number of files to create 0->(-tf)-1.')

    # Extrapolation mode
    parser.add_argument('-e', '--extrapolation', type=int, default=0, help='Extrapolation method: 0 -> probabilistic method, 1 -> trilineal cilindric interpolation, 2 -> method.')

    # Initial time step to process
    parser.add_argument('-dti', '--dT_initial', type=int, default=None, help='Initial time step to process.')
    
    # Final time step to process
    parser.add_argument('-dtf', '--dT_final', type=int, default=None, help='Final time step to process.')

    # Mode to process the files
    parser.add_argument('-m', '--mode', type=int, default=0, help='Mode: 0 = 0 -> t, 1 = t_initial -> t, 2 = t_initial -> t_final.')

    # Mode to compute the smoothing length
    parser.add_argument('-hm', '--smoothig_length_mode', type=int, default=0, help='Mode: 0 = density based, 1 = adaptative, 2 .')

    # Mode to compute the functions
    parser.add_argument('-vm', '--vectorized_mode', type=int, default=0, help='Mode: 0 = no vectorized, 1 = vectorized.')

    # Mode for compute the dust
    parser.add_argument('-dm', '--dust_mode', type=int, default=0, help='Mode: 0 = no dust, 1 = dust.')
    
    args = parser.parse_args()
    total_cpus = args.processors
    output_dir = args.output
    path_outputs_fargo = args.output_fargo
    total_timesteps = args.times
    Ntot = args.particles
    alpha = args.alpha
    beta = args.beta
    eta = float(args.eta)
    extrapolation_mode = args.extrapolation
    total_files = int(args.total_files)
    dT_initial = args.dT_initial
    dT_final = args.dT_final
    mode = args.mode
    h_mode = args.smoothig_length_mode
    vectorized_mode = args.vectorized_mode
    dust_mode = args.dust_mode

    if mode == 0:
        dT_initial = None
        dT_final = None
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, eta, extrapolation_mode, total_files, h_mode, vectorized_mode, mode, dust_mode, args)
        
    elif mode == 1:
        dT_final = None
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, eta, extrapolation_mode, total_files, h_mode, vectorized_mode, mode, dust_mode, args, dT_initial)
        
    else:
        main(total_cpus, output_dir, path_outputs_fargo, total_timesteps, Ntot, alpha, beta, eta, extrapolation_mode, total_files, h_mode, vectorized_mode, mode, dust_mode, args, dT_initial, dT_final)
        
    
    print(f"Tiempo total: {round(time.time() - init_time, 5)} segundos.")


        
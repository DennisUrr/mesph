import time
import numpy as np
import os
from multiprocessing import cpu_count
from utils.parameters import read_parameters
from utils.conversions import to_cartesian_3d, velocities_to_cartesian_3d
from utils.sampling import sample_particles_3d_partial, assign_particle_velocities_from_grid_3d_partial, assign_particle_densities_from_grid_3d_partial, assign_particle_internal_energies_from_grid_3d_partial, sample_particles_3d_interpolated
from utils.physics import compute_pressure, compute_acceleration_3d_partial, compute_artificial_viscosity_3d_partial, compute_particle_mass_3d
from utils.sph_utils import compute_smoothing_length_3d
from utils.hdf5_utils import create_snapshot_file, create_unique_directory, copy_files_to_directory

from tqdm import tqdm
import argparse


def main():
    dT=str(0)

    global particle_mass
    global total_timesteps
    global params, gamma, ASPECTRATIO, alpha, beta, epsilon, total_cpus, Ntot, Ntot_per_file, phi, r, theta, phimed, rmed, thetamed, nphi, nr, ntheta

    path_outputs_fargo = '../FARGO3D/public/outputs/p3disof/'
    output_dir = 'outputs/snapshot'
    unique_dir = create_unique_directory(output_dir)
    total_timesteps = 2  # Cambiar según tus datos
    Ntot = 20000  # Número total de partículas

    params = read_parameters( path_outputs_fargo + "/variables.par")
    gamma = float(params['GAMMA'])
    ASPECTRATIO = float(params['ASPECTRATIO'])
    alpha = 0.6
    beta = 1
    epsilon = 1e-5
    total_cpus = 2
    Ntot_per_file = Ntot // total_cpus  # Número de partículas por archivo
    base_filename = 'snapshot_3d'

    # Carga el dominio angular y radial de FARGO3D
    phi = np.loadtxt( path_outputs_fargo + "/domain_x.dat")
    r = np.loadtxt( path_outputs_fargo + "domain_y.dat")[3:-4]
    theta = np.loadtxt( path_outputs_fargo + "domain_z.dat")[3:-3]


    # Calcula los puntos medios para el ángulo y el radio
    phimed = 0.5*(phi[1:]+phi[:-1])
    rmed   = 0.5*(r[1:]+r[:-1])
    thetamed = 0.5*(theta[1:]+theta[:-1])

    # Tamaños de las matrices para phi y r
    nphi = len(phimed)
    nr   = len(rmed)
    ntheta = len(theta)


    for dT in tqdm(range(total_timesteps), desc='Progreso total de pasos de tiempo'):
        dT = str(dT)
        start_time = time.time()
        # Aquí necesitas cargar o calcular los datos para cada paso de tiempo
        # Carga la densidad del gas
        rho = np.fromfile( path_outputs_fargo + '/gasdens' + dT + '.dat').reshape(len(theta)-1,len(r),len(phi)-1)#volume density

        # Carga las velocidades del gas

        vphi = np.fromfile( path_outputs_fargo + '/gasvx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vr = np.fromfile( path_outputs_fargo + '/gasvy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
        vtheta = np.fromfile( path_outputs_fargo + '/gasvz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        print("vphi.shape: ", vphi.shape)
        print("vr.shape: ", vr.shape)
        print("vtheta.shape: ", vtheta.shape)

        # Cargar las energías internas del gas en 3D
        u = np.fromfile( path_outputs_fargo + '/gasenergy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

        print("Cargando datos de FARGO3D...")
        print(f"tomó {round(time.time() - start_time, 5)} segundos.")

        for file_idx in tqdm(range(total_cpus), desc=f'Procesando archivos para el tiempo {dT}', leave=False):
            start_idx = file_idx * Ntot_per_file
            end_idx = start_idx + Ntot_per_file
            if file_idx == total_cpus - 1:  # Ajustar para el último archivo
                end_idx = Ntot
                    
            print("==============================================================")
            print("#####                    TIEMPO = " + dT + "                       #####")
            print("#####                    ARCHIVO = " + str(file_idx) + "                      #####")
            print("==============================================================")
            
            start_time = time.time()

            # 1. Muestrear partículas
            # Muestrear partículas para el subconjunto actual
            #rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            rlist, philist, thetalist = sample_particles_3d_interpolated(rho, rmed, phimed, thetamed, start_idx, end_idx)

            # # Convertir posiciones de las partículas a coordenadas cartesianas 3D
            x, y, z = to_cartesian_3d(rlist, philist, thetalist)
            positions_3d = np.column_stack((x, y, z))

            print("Muestreo de partículas completado.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")
            
            # Convertir velocidades de polares a cartesianas
            start_time = time.time()
            vrlist, vphilist, vthetalist = assign_particle_velocities_from_grid_3d_partial(vphi, vr, vtheta, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
            print("vrlist.shape: ", vrlist.shape)
            print("vphilist.shape: ", vphilist.shape)
            print("vthetalist.shape: ", vthetalist.shape)
            vx, vy, vz = velocities_to_cartesian_3d(vrlist, vphilist, vthetalist, rlist, philist, thetalist)
            velocities = np.column_stack((vx, vy, vz))
            print("Velocidades de partículas asignadas.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

            # 2. Asignar propiedades a las partículas
            start_time = time.time()
            densities = assign_particle_densities_from_grid_3d_partial(rho, rlist, philist, thetalist, rmed, phimed, thetamed, start_idx, end_idx)
            particle_energies = assign_particle_internal_energies_from_grid_3d_partial(rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx)
            print("Densidades de partículas asignadas.")
            print("Dimensión densities: ", densities.shape)
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")


            # 3. Calcular masa de las partículas
            start_time = time.time()
            particle_mass = compute_particle_mass_3d(nr, ntheta, rho, ASPECTRATIO, params, Ntot)
            masses = np.full(Ntot_per_file, particle_mass, dtype=np.float32)
            print("Masas de partículas asignadas.")
            print("Dimensión masses: ", masses.shape)
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

            # # Calcular la presión de las partículas
            start_time = time.time()
            pressures = compute_pressure(densities, particle_energies, gamma)
            print("Presiones de partículas asignadas.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

            # 4. Calcular longitud de suavizado adaptativa
            h_values = compute_smoothing_length_3d(masses, densities)
            print("Longitudes de suavizado adaptativas calculadas.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

            # 5. Calcular viscosidad artificial
            start_time = time.time()
            viscosities = compute_artificial_viscosity_3d_partial(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)
            print("Viscosidades de partículas calculadas.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")
            # 6. Calcular aceleración
            start_time = time.time()
            accelerations = compute_acceleration_3d_partial(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
            print("Aceleraciones de partículas calculadas.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

            ids = np.arange(start_idx, end_idx, dtype=np.int32)
            start_time = time.time()
            create_snapshot_file(dT, file_idx, Ntot_per_file, positions_3d, velocities, ids, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_cpus, unique_dir)
            print("Archivo HDF5 creado.")
            print(f"tomó {round(time.time() - start_time, 5)} segundos.")

    source_files = [
    'outputs/splash.defaults',  
    'outputs/splash.limits'   
    ]

    destination_directory = 'outputs/' + unique_dir

    # Asegúrate de que el directorio destino existe
    if not os.path.exists(destination_directory):
        print(f"El directorio {destination_directory} no existe.")

    # Copia los archivos
    copy_files_to_directory(source_files, destination_directory)

if __name__ == '__main__':
    

    init_time = time.time()
    main()
    print(f"Tiempo total: {round(time.time() - init_time, 5)} segundos.")
    



    
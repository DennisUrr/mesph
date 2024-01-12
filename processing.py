import numpy as np
import traceback
from utils.conversions import spherical_to_cartesian, velocities_to_cartesian_3d, to_cartesian
from utils.sampling import sample_particles_3d_partial, assign_particle_velocities_from_grid_3d_partial, assign_particle_densities_from_grid_3d_partial, assign_particle_internal_energies_from_grid_3d_partial, sample_particles_3d_trilineal_partial, sample_particle_velocities, calculate_velocity_probability_matrix
from utils.physics import compute_pressure, compute_particle_mass_3d, compute_artificial_viscosity_3d_partial_vectorized, compute_acceleration_3d_partial_vectorized, compute_artificial_viscosity_3d_partial, compute_acceleration_3d_partial
from utils.sph_utils import compute_smoothing_length_density_based, compute_adaptive_smoothing_length_adaptative, compute_smoothing_length_neighbors_based
from utils.hdf5_utils import create_snapshot_file

def process_file(file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, extrapolation_mode, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, unique_dir, total_files, h_mode, vectorized_mode, base_filename = 'snapshot_3d'): 
    
    try:
        # Sampling the particles for the actual subset
        if extrapolation_mode == 0:
            rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            x, y, z = to_cartesian(rlist, philist, thetalist)
        elif extrapolation_mode == 1:
            rlist, philist, thetalist = sample_particles_3d_trilineal_partial(rho, phi, theta, r, start_idx, end_idx)
            x, y, z = to_cartesian(rlist, philist, thetalist)
        else:
            rlist, philist, thetalist = sample_particles_3d_partial_1(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            x, y, z = to_cartesian(rlist, philist, thetalist)
        
        # Convert spherical coordinates to Cartesian coordinates
        positions_3d = np.column_stack((x, y, z))
        
        # Convert velocity components from spherical to Cartesian coordinates
        probability_matrix = calculate_velocity_probability_matrix(vphi, vr, vtheta)
        vrlist, vphilist, vthetalist = sample_particle_velocities(probability_matrix, vphi, vr, vtheta, r, phi, theta, rmed, phimed, thetamed, start_idx, end_idx)
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
        
        if h_mode == 0:
            
            h_values = compute_smoothing_length_density_based(masses, densities)
        elif h_mode == 1:
            
            h_values = compute_adaptive_smoothing_length_adaptative(positions_3d, densities)
        elif h_mode == 2:
            
            h_values = compute_smoothing_length_neighbors_based(positions_3d)
        
        if vectorized_mode == 1:
            # Compute viscosities
            viscosities = compute_artificial_viscosity_3d_partial_vectorized(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)
            # Compute accelerations    
            accelerations = compute_acceleration_3d_partial_vectorized(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
        elif vectorized_mode == 0:
            # Compute viscosities
            viscosities = compute_artificial_viscosity_3d_partial(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)
            # Compute accelerations
            accelerations = compute_acceleration_3d_partial(positions_3d, densities, pressures, particle_mass, h_values, viscosities)

        # Create snapshot file
        create_snapshot_file(dT, file_idx, Ntot_per_file, positions_3d, velocities, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, base_filename, total_files, unique_dir, start_idx, end_idx)

    except Exception as e:
        print(f"Error processing file {file_idx} at time step {dT}: {e}")
        traceback.print_exc()
        raise e

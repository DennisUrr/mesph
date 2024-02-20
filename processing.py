import numpy as np
import traceback
from utils.conversions import velocities_to_cartesian_3d, to_cartesian
from utils.sampling import sample_particles_3d_trilineal_partial, sample_particles_3d_with_velocity_density, interpolate_velocities, sample_particles_3d_partial, interpolate_densities, interpolate_internal_energies
from utils.physics import compute_pressure, compute_particle_mass_3d, compute_artificial_viscosity_3d_partial_vectorized, compute_acceleration_3d_partial_vectorized, compute_artificial_viscosity_3d_partial, compute_acceleration_3d_partial
from utils.sph_utils import compute_smoothing_length_density_based, compute_adaptive_smoothing_length_adaptative, compute_smoothing_length_neighbors_based
from utils.hdf5_utils import write_to_file, write_to_file_dust

def process_file(file_idx, dT, params, gamma, ASPECTRATIO, alpha, beta, eta, extrapolation_mode, Ntot, Ntot_per_file, rho, phi, theta, r, phimed, rmed, thetamed, vphi, vr, vtheta, u, nr, ntheta, start_idx, end_idx, h_mode, vectorized_mode, dust_mode, FRAME, OMEGAFRAME, rho_dust=None, vr_dust=None, vphi_dust=None, vtheta_dust=None): 
    """
    Processes a single simulation file, sampling particle properties, converting coordinates,
    calculating velocities, pressures, smoothing lengths, viscosities, and accelerations for
    both gas and dust particles, depending on the mode.

    Parameters:
    - file_idx: Index of the file being processed.
    - dT: Current timestep of the simulation.
    - params: Simulation parameters.
    - gamma: Adiabatic index.
    - ASPECTRATIO, alpha, beta, eta: Simulation parameters for viscosity, smoothing length, etc.
    - extrapolation_mode: Determines the method for sampling and interpolating properties.
    - Ntot, Ntot_per_file: Total number of particles and particles per file.
    - rho, phi, theta, r: Density and spherical coordinates grids for the simulation.
    - phimed, rmed, thetamed: Median values of the spherical coordinates.
    - vphi, vr, vtheta: Velocity components in spherical coordinates.
    - u: Internal energy grid.
    - nr, ntheta: Number of radial and theta grid points.
    - start_idx, end_idx: Indices for processing subsets of particles.
    - h_mode, vectorized_mode, dust_mode: Modes for calculating smoothing lengths, enabling vectorization, and handling dust.
    - FRAME, OMEGAFRAME: Reference frame parameters.
    - rho_dust, vr_dust, vphi_dust, vtheta_dust: Dust density and velocity components (optional).

    Returns:
    Arrays of positions, velocities, masses, internal energies, densities, smoothing lengths,
    accelerations, and pressures for gas and optionally for dust particles.

    Raises:
    Exception with error message and traceback if processing fails.
    """
    try:
        # Sampling the particles and interpolating their properties based on the mode
        if extrapolation_mode == 0:
            # Direct sampling for a subset of particles including velocity and density
            rlist, philist, thetalist, vrlist, vphilist, vthetalist, densities = sample_particles_3d_with_velocity_density(rho, phi, theta, rmed, phimed, thetamed, r, vphi, vr, vtheta, start_idx, end_idx)
        else:
            # Separate sampling and interpolation for particle positions and velocities
            rlist, philist, thetalist = sample_particles_3d_partial(rho, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            vrlist, vphilist, vthetalist = interpolate_velocities(vr, vphi, vtheta, r, phi, theta, rlist, philist, thetalist)
            # Interpolates densities instead of direct sampling
            densities = interpolate_densities(rho, r, phi, theta, rlist, philist, thetalist)
        
        # Converts spherical to Cartesian coordinates for position and velocity
        x, y, z = to_cartesian(rlist, philist, thetalist)
        vx, vy, vz = velocities_to_cartesian_3d(vrlist, vphilist, vthetalist, rlist, philist, thetalist)
        positions_3d = np.column_stack((z, y, x))
        velocities = np.column_stack((vx, vy, vz))
        
        # Assigns internal energies and computes particle masses
        #particle_energies = assign_particle_internal_energies_from_grid_3d_partial(rlist, philist, thetalist, u, rmed, phimed, thetamed, start_idx, end_idx)
        particle_energies = interpolate_internal_energies(u, r, phi, theta, rlist, philist, thetalist)
        particle_mass = compute_particle_mass_3d(nr, ntheta, rho, ASPECTRATIO, params, Ntot)
        masses = np.full(Ntot_per_file, particle_mass, dtype=np.float32)

        # Calculates pressures for the particles
        pressures = compute_pressure(densities, particle_energies, gamma)
        
        # Determines smoothing lengths based on the selected mode
        if h_mode == 0:
            h_values = compute_smoothing_length_density_based(masses, densities, eta)
        elif h_mode == 1:
            h_values = compute_adaptive_smoothing_length_adaptative(positions_3d, densities)
        elif h_mode == 2:
            h_values = compute_smoothing_length_neighbors_based(positions_3d)
        
        # Calculates viscosities and accelerations, using vectorized calculations if enabled
        if vectorized_mode == 1:
            viscosities = compute_artificial_viscosity_3d_partial_vectorized(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)
            accelerations = compute_acceleration_3d_partial_vectorized(positions_3d, densities, pressures, particle_mass, h_values, viscosities)
        else:
            viscosities = compute_artificial_viscosity_3d_partial(positions_3d, vx, vy, vz, densities, particle_energies, h_values, alpha, beta)
            accelerations = compute_acceleration_3d_partial(positions_3d, densities, pressures, particle_mass, h_values, viscosities)

        # Handles dust particles if dust mode is enabled
        if dust_mode == 1:
            # Sampling and interpolating for dust particles
            rlist_dust, philist_dust, thetalist_dust = sample_particles_3d_partial(rho_dust, phi, theta, rmed, phimed, thetamed, r, start_idx, end_idx)
            vrlist_dust, vphilist_dust, vthetalist_dust = interpolate_velocities(vr_dust, vphi_dust, vtheta_dust, r, phi, theta, rlist_dust, philist_dust, thetalist_dust, FRAME, OMEGAFRAME)
            densities_dust = interpolate_densities(rho_dust, r, phi, theta, rlist_dust, philist_dust, thetalist_dust)
            x_dust, y_dust, z_dust = to_cartesian(rlist_dust, philist_dust, thetalist_dust)
            vx_dust, vy_dust, vz_dust = velocities_to_cartesian_3d(vrlist_dust, vphilist_dust, vthetalist_dust, rlist_dust, philist_dust, thetalist_dust)
            positions_3d_dust = np.column_stack((z_dust, y_dust, x_dust))
            velocities_dust = np.column_stack((vx_dust, vy_dust, vz_dust))
            # Returns both gas and dust particle properties if dust mode is enabled
            return positions_3d, velocities, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, positions_3d_dust, velocities_dust, densities_dust

        # Returns gas particle properties
        return positions_3d, velocities, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities
    
    except Exception as e:
        # Handles exceptions by printing an error message and stack trace, then re-raises the exception
        print(f"Error processing file {file_idx} at time step {dT}: {e}")
        traceback.print_exc()
        raise e


def combine_and_write_results(results, unique_dir, dT, total_files, dust_mode):
    # Para gas
    for file_idx in range(total_files):
        file_results = [res for res in results]
        if dust_mode == 0:
            combined_results = combine_subsets(file_results)
            write_to_file(dT, file_idx, combined_results, unique_dir, total_files)
        else:
            #print("Dust mode")
            # Para polvo, maneja los resultados del polvo de manera separada
            combined_results_dust = combine_dust_subsets(file_results)
            write_to_file_dust(dT, file_idx, combined_results_dust, unique_dir, total_files)
            
            # Además, manejar y escribir los resultados del gas si se requiere
            combined_results_gas = combine_subsets(file_results)
            write_to_file(dT, file_idx, combined_results_gas, unique_dir, total_files)


def combine_subsets(file_results):
    """
    Combines the subsets of simulation results for gas particles into single arrays. This function is
    used to aggregate results from multiple processing steps or files into a unified set of data arrays
    for positions, velocities, masses, particle energies, densities, smoothing lengths, accelerations,
    pressures, and viscosities.

    Parameters:
    - file_results: A list of tuples, each containing arrays of particle properties from individual
                    processing steps or files.

    Returns:
    A tuple of numpy arrays containing combined properties for all processed gas particles:
    - combined_positions: 3D positions of particles.
    - combined_velocities: Velocities of particles.
    - combined_masses: Masses of particles.
    - combined_particle_energies: Internal energies of particles.
    - combined_densities: Densities of particles.
    - combined_h_values: Smoothing lengths of particles.
    - combined_accelerations: Accelerations of particles.
    - combined_pressures: Pressures of particles.
    - combined_viscosities: Viscosities of particles.
    """

    # Initialize empty lists to collect combined properties
    combined_positions = []
    combined_velocities = []
    combined_masses = []
    combined_particle_energies = []
    combined_densities = []
    combined_h_values = []
    combined_accelerations = []
    combined_pressures = []
    combined_viscosities = []

    # Iterate through each set of results and extend the combined lists
    for result in file_results:
        # Unpack the result tuple and ignore additional dust properties, if present
        positions, velocities, masses, particle_energies, densities, h_values, accelerations, pressures, viscosities, *_ = result

        combined_positions.extend(positions)
        combined_velocities.extend(velocities)
        combined_masses.extend(masses)
        combined_particle_energies.extend(particle_energies)
        combined_densities.extend(densities)
        combined_h_values.extend(h_values)
        combined_accelerations.extend(accelerations)
        combined_pressures.extend(pressures)
        combined_viscosities.extend(viscosities)

    # Convert the combined lists into numpy arrays for efficient handling
    combined_positions = np.array(combined_positions)
    combined_velocities = np.array(combined_velocities)
    combined_masses = np.array(combined_masses)
    combined_particle_energies = np.array(combined_particle_energies)
    combined_densities = np.array(combined_densities)
    combined_h_values = np.array(combined_h_values)
    combined_accelerations = np.array(combined_accelerations)
    combined_pressures = np.array(combined_pressures)
    combined_viscosities = np.array(combined_viscosities)

    return (combined_positions, combined_velocities, combined_masses, combined_particle_energies, combined_densities, combined_h_values, combined_accelerations, combined_pressures, combined_viscosities)

def combine_dust_subsets(file_results):
    """
    Combines the subsets of simulation results for dust particles into single arrays, similar to the
    combine_subsets function but specifically for dust. This function aggregates results from multiple
    processing steps or files into unified data arrays for positions, velocities, masses, and densities
    of dust particles.

    Parameters:
    - file_results: A list of tuples, each containing arrays of dust particle properties from individual
                    processing steps or files.

    Returns:
    A tuple of numpy arrays containing combined properties for all processed dust particles:
    - combined_positions: 3D positions of dust particles.
    - combined_velocities: Velocities of dust particles.
    - combined_masses: Masses of dust particles.
    - combined_densities: Densities of dust particles.
    - combined_h_values: Smoothing lengths of dust particles.
    """

    # Initialize empty lists to collect combined properties for dust particles
    combined_positions = []
    combined_velocities = []
    combined_masses = []
    combined_densities = []
    combined_h_values = []

    # Iterate through each set of results and extend the combined lists for dust properties
    for result in file_results:
        # Unpack the result tuple, focusing on dust properties
        _, _, masses, _, _, h_values, _, _, _, positions, velocities, densities = result

        combined_positions.extend(positions)
        combined_velocities.extend(velocities)
        combined_masses.extend(masses)
        combined_densities.extend(densities)
        combined_h_values.extend(h_values)

    # Convert the combined lists into numpy arrays for efficient handling and analysis
    combined_positions = np.array(combined_positions)
    combined_velocities = np.array(combined_velocities)
    combined_masses = np.array(combined_masses)
    combined_densities = np.array(combined_densities)
    combined_h_values = np.array(combined_h_values)

    return combined_positions, combined_velocities, combined_masses, combined_densities, combined_h_values

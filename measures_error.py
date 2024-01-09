import numpy as np
import argparse
from utils.conversions import to_spherical_velocity, to_cartesian, to_spherical
from scipy.interpolate import RegularGridInterpolator

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to cartesian coordinates (x, y, z).
    """
    x = r[:, np.newaxis, np.newaxis] * np.sin(theta)[np.newaxis, :, np.newaxis] * np.cos(phi)[np.newaxis, np.newaxis, :]
    y = r[:, np.newaxis, np.newaxis] * np.sin(theta)[np.newaxis, :, np.newaxis] * np.sin(phi)[np.newaxis, np.newaxis, :]
    z = r[:, np.newaxis, np.newaxis] * np.cos(theta)[np.newaxis, :, np.newaxis]
    return x, y, z


def calculate_mae(original_data, converted_data):
    """
    Calculates the Mean Absolute Error (MAE) between two data sets.

    Args:
    original_data (numpy.ndarray): Original data set.
    converted_data (numpy.ndarray): Converted data set.

    Returns:
    float: MAE value.
    """
    absolute_difference = np.abs(original_data - converted_data)
    mae = np.mean(absolute_difference)
    return mae

def calculate_rmse(original_data, converted_data):
    """
    Calculates the Root Mean Square Error (RMSE) between two data sets.

    Args:
    original_data (numpy.ndarray): Original data set.
    converted_data (numpy.ndarray): Converted data set.

    Returns:
    float: RMSE value.
    """
    squared_difference = (original_data - converted_data) ** 2
    rmse = np.sqrt(np.mean(squared_difference))
    return rmse


def load_data_fargo(path_outputs_fargo, dT):
    phi = np.loadtxt(path_outputs_fargo + "domain_x.dat")
    r = np.loadtxt(path_outputs_fargo + "domain_y.dat")[3:-4]
    theta = np.loadtxt(path_outputs_fargo + "domain_z.dat")[3:-3]

    rho = np.fromfile(path_outputs_fargo + 'gasdens' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1) # volume density
            
    # Load the gas velocities
    vphi = np.fromfile(path_outputs_fargo + 'gasvx' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
    vr = np.fromfile(path_outputs_fargo + 'gasvy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
    vtheta = np.fromfile(path_outputs_fargo + 'gasvz' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)
            
    # Load the gas internal energy
    u = np.fromfile(path_outputs_fargo + 'gasenergy' + dT + '.dat').reshape(len(theta)-1, len(r), len(phi)-1)

    return phi, r, theta, rho, vphi, vr, vtheta, u

def load_data_mesph(path_outputs_mesph, dT):

    with open(path_outputs_mesph + "snapshot_3d_00" + dT + ".0.hdf5.gridstream", "rb") as file:
        nx, ny, nz, ncolumns = np.fromfile(file, dtype=np.int32, count=4)
        time, xmin, xmax, ymin, ymax, zmin, zmax = np.fromfile(file, dtype=np.float64, count=7)
        #print(f"nx: {nx}, ny: {ny}, nz: {nz}, ncolumns: {ncolumns}")
        #print(" nx * ny * nz = ",  nx * ny * nz)
        rho = np.fromfile(file, dtype=np.float64, count = (nx * ny * nz))
        #print(f"rho.shape: {rho.shape}")
    rho_array = rho.reshape((nx, ny, nz))
    #print(f"rho_array.shape: {rho_array.shape}")
    return rho_array, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parallel particle processing for astrophysical simulations.')

    parser.add_argument('-of', '--output_fargo', type=str, default='../public/outputs/dust3d/', help='Directory containing FARGO3D output files.')
    parser.add_argument('-om', '--output_mesph', type=str, default='../mesph-remote/outputs/snapshot_p30_n200000_a0.6_b1_tf30_e0_dti90_dtf100_m2_hm0_vm0/', help='Directory containing MESPHRAY output files.')
    parser.add_argument('-dT', '--time_step', type=str, default='99', help='Time step for data extraction.')

    args = parser.parse_args()

    path_outputs_fargo = args.output_fargo
    path_outputs_mesph = args.output_mesph
    dT = args.time_step

    phi, r, theta, rho, vphi, vr, vtheta, u = load_data_fargo(path_outputs_fargo, dT)
    #print(f"phi.shape: {phi.shape}, r.shape: {r.shape}, theta.shape: {theta.shape}, rho.shape: {rho.shape}, vphi.shape: {vphi.shape}, vr.shape: {vr.shape}, vtheta.shape: {vtheta.shape}, u.shape: {u.shape}")
    rho_array, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz = load_data_mesph(path_outputs_mesph, '8')

    print(rho_array.shape)

    x, y, z = spherical_to_cartesian(r, theta, phi)
    
    x_cartesian = np.linspace(xmin, xmax, nx)
    y_cartesian = np.linspace(ymin, ymax, ny)
    z_cartesian = np.linspace(zmin, zmax, nz)

    interpolator = RegularGridInterpolator((theta[:-1], r, phi[:-1]), rho, bounds_error=False, fill_value=None)

    cartesian_points = np.array(np.meshgrid(x_cartesian, y_cartesian, z_cartesian)).T.reshape(-1, 3)

    rho_interpolated = interpolator(cartesian_points).reshape(nx, ny, nz)

    rmse = calculate_rmse(rho_interpolated, rho_array)

    print(f"RMSE: {rmse}")
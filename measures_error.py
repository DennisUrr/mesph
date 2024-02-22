import numpy as np
import argparse
from utils.conversions import to_spherical_velocity, to_cartesian, to_spherical
from scipy.interpolate import RegularGridInterpolator
import trilinear
from utils.parameters import read_parameters
from run_splash import generate_shell_script_measures_errors

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

    parser.add_argument('-of', '--output_fargo', type=str, default='outputs_fargo/', help='Directory containing FARGO3D output files.')
    parser.add_argument('-om', '--output_mesph', type=str, default='outputs/snapshot_p20_n1000000_a0.6_b1_eta1.0_tf8_e1_dti130_m1_hm0_vm0_dm1/', help='Directory containing MESPHRAY output files.')
    parser.add_argument('-dT', '--time_step', type=str, default='131', help='Time step for data extraction.')

    args = parser.parse_args()

    path_outputs_fargo = args.output_fargo
    path_outputs_mesph = args.output_mesph
    dT = args.time_step


    phi, r, theta, rho, vphi, vr, vtheta, u = load_data_fargo(path_outputs_fargo, dT)
    #print(f"phi.shape: {phi.shape}, r.shape: {r.shape}, theta.shape: {theta.shape}, rho.shape: {rho.shape}, vphi.shape: {vphi.shape}, vr.shape: {vr.shape}, vtheta.shape: {vtheta.shape}, u.shape: {u.shape}")
    rho_array, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz = load_data_mesph(path_outputs_mesph, '1')

    x, y, z = spherical_to_cartesian(r, theta, phi)
    
    x_cartesian = np.linspace(xmin, xmax, nx)
    y_cartesian = np.linspace(ymin, ymax, ny)
    z_cartesian = np.linspace(zmin, zmax, nz)


    rho_interpolated_1 = trilinear.trilinear(rho, r, phi[:-1], theta[:-1], z_cartesian, y_cartesian, x_cartesian, np.min(rho))

    mae = calculate_mae(rho_array, rho_interpolated_1)
    rmse= calculate_rmse(rho_array, rho_interpolated_1)


    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")




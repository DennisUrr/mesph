
import os


def run_splash(total_timesteps, output_dir, dust_mode):
    # Siempre generar el script para gas
    base_filename_gas = "snapshot_3d"
    shell_script_name_gas = "run_splash_gas.sh"
    filenames_gas = [f'{base_filename_gas}_{t:03d}.0.hdf5' for t in range(total_timesteps)]
    generate_shell_script(output_dir, shell_script_name_gas, filenames_gas)

    # Generar el script para polvo solo si dust_mode está activo
    if dust_mode == 1:
        base_filename_dust = "snapshot_3d_dust"
        shell_script_name_dust = "run_splash_dust.sh"
        filenames_dust = [f'{base_filename_dust}_{t:03d}.0.hdf5' for t in range(total_timesteps)]
        generate_shell_script(output_dir, shell_script_name_dust, filenames_dust)

def generate_shell_script(output_dir, script_name, filenames):
    # Construye el comando completo
    command = ["splash", "-f", "gadget_hdf5"] + filenames

    # Genera el script shell
    
    with open(script_name, "w") as file:
        file.write("#!/bin/bash\n")
        file.write("cd " + output_dir + "\n")
        file.write(' '.join(command) + "\n")

    # Hace el script ejecutable
    os.chmod(script_name, 0o755)

def generate_shell_script_measures_errors(output_dir, NX, NY, NZ):

    base_filename = "snapshot_3d"
    script_name = "run_measures_error.sh"
    filenames = [f'{base_filename}_{t:03d}.0.hdf5' for t in range(2)]
    

    # Construye el comando completo
    command = ["splash", "to gridstream", f'--npix={NX},{NY},{NZ}'] + filenames

    # Genera el script shell
    
    with open(script_name, "w") as file:
        file.write("#!/bin/bash\n")
        file.write("cd " + output_dir + "\n")
        file.write(' '.join(command) + "\n")

    # Hace el script ejecutable
    os.chmod(script_name, 0o755)
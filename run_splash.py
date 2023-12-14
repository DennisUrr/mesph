
import os

def run_splash(total_timesteps, output_dir):
    base_filename = "snapshot_3d"

    # Construye los nombres de archivo
    filenames = [f'{base_filename}_{t:03d}.0.hdf5' for t in range(total_timesteps)]

    # Construye el comando completo
    command = ["splash", "-f", "gadget_hdf5"] + filenames

    # Genera el script shell
    shell_script_name = "run_splash.sh"
    with open(shell_script_name, "w") as file:
        file.write("#!/bin/bash\n")
        file.write("cd " + output_dir + "\n")
        file.write(' '.join(command) + "\n")

    # Hace el script ejecutable
    os.chmod(shell_script_name, 0o755)
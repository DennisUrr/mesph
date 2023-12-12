
import os

def run_splash(total_timesteps, file_idx, output_dir):
    base_filename = "snapshot_3d"

    # Construye los nombres de archivo
    filenames = [f'{base_filename}_{t:03d}.0.hdf5' for t in range(total_timesteps)]

    # Construye el comando completo
    command = ["splash", "-f", "gadget_hdf5"] + filenames

    # Cambia al último directorio creado
    os.chdir(output_dir)

    print(command)
    # Ejecuta el comando
    #subprocess.run(command)
import h5py
import numpy as np

def validar_hdf5(nombre_archivo):
    with h5py.File(nombre_archivo, 'r') as f:
        print("Validando archivo:", nombre_archivo)

        # Validar estructura básica
        if "PartType0" not in f:
            print("Error: El grupo PartType0 no se encuentra en el archivo.")
            return
        if "Header" not in f:
            print("Error: El grupo Header no se encuentra en el archivo.")
            return

        pt0 = f["PartType0"]
        header = f["Header"]

        # Imprimir y validar atributos del header
        print("\nAtributos del Header:")
        for key, value in header.attrs.items():
            print(key + ":", value)
            if np.any(np.array(value) < 0):
                print("  Advertencia: El valor de", key, "es negativo.")

        # Imprimir y validar datasets de PartType0
        print("\nDatasets de PartType0:")
        for key, dataset in pt0.items():
            print(key + ":")
            print("  Forma:", dataset.shape)
            print("  Tipo de dato:", dataset.dtype)
            if key in ["Coordinates", "Velocities", "Masses", "SmoothingLength", "InternalEnergy"]:
                min_val, max_val = np.min(dataset), np.max(dataset)
                print("  Valor mínimo:", min_val)
                print("  Valor máximo:", max_val)
                if min_val < 0:
                    print("  Advertencia: Hay valores negativos en", key)
            if key == "ParticleIDs":
                if np.any(dataset[:] <= 0):
                    print("  Error: Hay IDs de partículas no positivos.")

        print("\nValidación completada.")

if __name__ == "__main__":
    nombre_archivo = "snapshot_000.0.hdf5"
    validar_hdf5(nombre_archivo)

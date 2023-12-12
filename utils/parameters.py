import os

def read_parameters(filename):
    """Read parameters from the given file."""
    params = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                # Si la línea contiene una tabulación, entonces se toma como una línea válida con un par clave-valor
                if "\t" in line:
                    key, value = line.strip().split("\t")
                    params[key.strip()] = value.strip()
        return params
    print("ERROR: The given file does not exist and it's required to run the program.")
    return {}
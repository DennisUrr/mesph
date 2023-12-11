# mesph

This code transalate the outputs of FARGO3D to a format that splash can read by extrapolating the data. Until the moment, this only translate to a GADGET-2 file with HDF5 format using two methods: a probabilistic one and a trilineal interpolation.

## Usage

The script `main_parallel.py` can be executed with several optional arguments to customize its behavior. It is designed to create multiple output files for each Fargo output file, with the number of output files being equal to the number of processors used.

### Arguments

- `-p` / `--processors`: Number of processors to use. Defaults to 2.
- `-o` / `--output`: Output directory for the generated files. Defaults to `outputs/snapshot`.
- `-of` / `--output_fargo`: Directory where FARGO3D outputs are located. Defaults to `../../public/outputs/p3disof/`.
- `-t` / `--times`: Number of time steps to transform. Defaults to 1.
- `-n` / `--particles`: Number of particles to transform. Defaults to 10000.
- `-a` / `--alpha`: Alpha parameter for artificial viscosity. Defaults to 0.6.
- `-b` / `--beta`: Beta parameter for artificial viscosity. Defaults to 1.
- `-e` / `--epsilon`: Epsilon parameter for artificial viscosity. Defaults to 1e-5.

### Running the Script

For instance, if you have 20 Fargo output files (like `domain.dat`, `gasdens{n}.dat`, `gasv{n}.dat`, `gasenergy{n}.dat`) and you want to use 8 processors, the script will generate 20*8 output files.

You can run the script with custom parameters like this:

```bash
python3 main_parallel.py -p 8 --t 20

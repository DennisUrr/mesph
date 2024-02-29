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
- `-tf` / `--total_files`: Number of files per snapshot. Defaults to 2.
- `-e` / `--extrapolation`: Extrapolation method: 0 -> probabilistic method, 1 -> trilinear interpolation. Defaults to 0'
- `-m` / `--mode`: Mode: 0 = 0 -> t, 1 = t_initial -> t, 2 = t_initial -> t_final. Defaults to 0.
- `-dti` / `--dT_initial`: Initial time step to process. Defaults to None.
- `-dtf` / `--dT_final`: Final time step to process. Defaults to None.
- `-hm` / `--smoothig_length_mode`: Mode: 0 = density based, 1 = adaptative, 2. Defaults to 0.
- `-vm` / `vectorized_mode`: Mode: 0 = no vectorized, 1 = vectorized. Defaults to 0. 
   
  
### Running the Script

For instance, if you have 20 Fargo output files (like `domain.dat`, `gasdens{n}.dat`, `gasv{n}.dat`, `gasenergy{n}.dat`) and you want to use 8 processors, the script will generate -tf number of output files. 

You can run the script with custom parameters like this:

```bash
python3 main.py -p 30 -t 20 -tf 8 -n 100000
```

Once its done, it will generate a executable file called run_splash_gas.sh that you can run on the command line. For this you need an X server if you want to use it interactive. 

```bash
./run_splash_gas.sh
```

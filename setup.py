from setuptools import setup, find_packages

setup(
    name="mesph",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.5",
        "h5py>=3.8.0",
        "scipy>=1.7.3",
        "matplotlib>=3.5.3",
    ],
    # metadata to display on PyPI
    author="Dennis Urrutia",
    author_email="dennis.urrutia@usach.cl",
    description="This is a program that transforms a set of outputs simulation files from FARGO3D into a SPH data, specifically to a GADGET-3 file with HDF5 format.",
    keywords="sph, eulerian mesh, fargo3d, gadget-3, hdf5, transformation, extrapolation",
    url="https://github.com/DennisUrr/mesph",
    license='LICENSE',
    scripts=['main.py', 'measures_error.py'],
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    include_package_data=True
)

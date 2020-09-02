from setuptools import setup


__version__ = '1.0.0'


with open("README.md", "r") as readme_file:
    readme = readme_file.read()


setup(
    name = 'pynerf',
    version = __version__,
    author = 'Adam Moyer',
    author_email = 'atom.moyer@gmail.com',
    description = 'A Numpy Implementation of the NeRF Algoritm for Global and Internal Molecular Coordinate Conversion',
    packages = ['nerf'],
    package_dir={'nerf' : 'nerf'},
    package_data={},
    install_requires = ['numpy'],
    include_package_data=True,
    zip_safe = False,
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/atom-moyer/nerf',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix'
    ],
)

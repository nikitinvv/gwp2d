from setuptools import setup, find_packages
setup(
    name='gwp2d',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    url='https://github.com/nikitinvv/gwp2d',
    packages=find_packages(),
    include_package_data = True,
    #scripts=[''],
    description='Gaussian wave-packet decompositon in 2D',
    zip_safe=False,
)

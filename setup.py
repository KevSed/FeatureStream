from setuptools import setup, find_packages

setup(name='features_from_phs',
      version='0.1',
      description='Generate features for analysis from photon stream data',
      url='http://github.com/KevSed/features_from_phs',
      author='Kevin Sedlaczek',
      author_email='kevin.sedlaczek@tu-dortmund.de',
      license='GNU',
      packages=['features_from_phs'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'print_progress'
          ],
      zip_safe=False
      )

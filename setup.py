from setuptools import setup, find_packages

setup(name='feature_stream',
      version='0.1',
      description='Generate features for analysis from photon stream data',
      url='http://github.com/KevSed/features_from_phs',
      author='Kevin Sedlaczek',
      author_email='kevin.sedlaczek@tu-dortmund.de',
      license='GNU',
      packages=['feature_stream'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas'
          ],
      zip_safe=False
      )

from setuptools import setup

setup(name='features_from_phs',
      version='0.1',
      description='Generate features for analysis from photon stream data',
      url='http://github.com/KevSed/features_from_phs',
      author='Kevin Sedlaczek',
      author_email='kevin.sedlaczek@tu-dortmund.de',
      license='MIT',
      packages=['features_from_phs,photon_stream,numpy,scipy,pandas,sys'],
      zip_safe=False)
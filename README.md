# FeatureStream

A python package to do several analysis tasks on [PhotonStream](https://github.com/fact-project/photon_stream) data, as proposed by the [FACT](https://github.com/fact-project) project.
This package provides functions for FACT data safed in the PhotonStream format. An open data sample of Crab observations can be found [here](https://fact-project.org/data/)

This package contains functions for
* image cleaning via DBSCAN
* image cleaning via pixel thresholds as is defined [here](https://github.com/fact-project/fact-tools)
* generating the usual Hillas features
* generating meta data

It can be used to generate this features and will return a dictionary containing those.

Installation can be done via git and pip:
```
$ git clone git@github.com:KevSed/FeatureStream.git
$ cd FeatureStream/
$ pip install .
```

Importing the package, loading a PhotonStream data file and cleaning an event using DBSCAN looks like this:
```python
import photon_stream as ps
from feature_stream import phs2image, cleaning

# read in a data file
reader = ps.EventListReader(data_file)

# get a first event
event = next(reader)

# get the list of arrival times per photon
lol = event.photon_stream.list_of_lists

# generate an image (number of photons for each of the 1440 pixlels in FACT)
image = phs2image(lol)

# do the DBSCAN cleaning and return a boolean array of all cleaned pixels
cleaned_pixel = cleaning(image, lol)

# same as above, but using the facttools cleaning
cleaned_pixel = facttools_cleaning(image, lol)
```

Generating all available features for a data or Monte Carlo file and saving them in a data frame:
```python
from feature_stream import gen_features

# Generating meta data and Hillas features and saving to data frame df
df = gen_features(data_file)
```

import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
from fact.io import to_h5py
from print_progress import print_progress

def gen_features(data_file, output_file):
    
    """ This generates a certain set of features from photon-stream data files that can be used for further analyses.
    
    Inputs:
    data_file:           location of input data file as string
    output_file as hdf5: destination of output data file as string
    
    """
    # data file
    reader = ps.EventListReader(data_file)
    
    # initialisation of list of dicts containing generated data
    events = list()

    # variables for print_progress
    j = 0
    le = 18017

    # loop for events
    for event in reader:
        j = j+1
        
        # safe x, y and t components of Photons. shape = (#photons,3)
        xyt = event.photon_stream.point_cloud
        x, y = xyt[:, :2].T

        # clustering of events
        clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream)  #, eps=0.07)

        # events from clusters
        mask = clustering.labels == 0

        if len(x[mask]) >= 1:

            # covariance and eigenvalues/vectors for later calculations
            cov = np.cov(x[mask], y[mask])
            eig_vals, eig_vecs = np.linalg.eigh(cov)

            # Descriptive statistics: mean, std dev, kurtosis, skewness
            cmean_x = np.mean(x)
            cmean_y = np.mean(y)
            cstd_x = np.std(x)
            cstd_y = np.std(y)
            ckurtosis_x = scipy.stats.kurtosis(x)
            ckurtosis_y = scipy.stats.kurtosis(y)
            cskewness_x = scipy.stats.skew(x)
            cskewness_y = scipy.stats.skew(y)

            # means of cluster
            cog_x = np.mean(x[mask])
            cog_y = np.mean(y[mask])

            # hillas parameter
            width, length = np.sqrt(eig_vals)
            delta = np.arctan2(eig_vecs[1, 1], eig_vecs[0, 1])
            angle = np.rad2deg(delta)

            # number of photons in biggest cluster
            biggest_cluster = np.argmax(np.bincount(clustering.labels[clustering.labels != -1]))
            size = len(x[clustering.labels == biggest_cluster])

            # number of clusters
            clusters = len(np.bincount(clustering.labels[clustering.labels != -1]))

            # put values in dict and append to list of dicts (events)
            ev = {'cog_x': cog_x, 'cog_y': cog_y, 'mean_x': cmean_x, 'stddev_x': cstd_x, 'stddev_y': cstd_y, 'mean_y': cmean_y, 'width': width, 'length': length, 'angle': angle, 'kurtosis_x': ckurtosis_x, 'kurtosis_y': ckurtosis_y, 'skewness_x': cskewness_x, 'skewness_y': cskewness_y, 'clusters': clusters, 'size': size}
            events.append(ev)
            
        print_progress(j + 1, le)
    
    # save list of dicts in pandas data frame        
    df = pd.DataFrame(events)
    
    # save data frame in output hdf5-file
    to_h5py(output_file, df, key='events')

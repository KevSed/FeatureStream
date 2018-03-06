import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
import warnings
from fact.instrument import camera_distance_mm_to_deg


def gen_data_features(data_file):

    """ This generates a certain set of features from photon-stream data files
        that can be used for further analyses.

    Inputs:
    data_file:           location of input data file as string

    return:
    pandas data frame with features

    """

    # read only data
    reader = ps.EventListReader(data_file)

    # initialisation of list of dicts containing generated data
    events = list()

    # loop for events
    for event in reader:
        if event.observation_info.trigger_type != 4:
            continue

        # safe x, y and t components of Photons. shape = (#photons,3)
        xyt = event.photon_stream.point_cloud
        x, y = xyt[:, :2].T
        x = np.rad2deg(x) / camera_distance_mm_to_deg(1)
        y = np.rad2deg(y) / camera_distance_mm_to_deg(1)

        # clustering of events
        clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream)

        # events from clusters
        mask = clustering.labels == 0

        # empty dict for values
        ev = {}

        if len(x[mask]) >= 1:
            # observation pointing
            ev['pointing_position_az'] = event.az
            ev['pointing_position_zd'] = event.zd

            # Meta information on event [ONLY DATA]
            ev['run'] = event.observation_info.run
            ev['event'] = event.observation_info.event
            ev['night'] = event.observation_info.night
            ev['timestamp'] = event.observation_info.time

            # covariance and eigenvalues/vectors for later calculations
            cov = np.cov(x[mask], y[mask])
            eig_vals, eig_vecs = np.linalg.eigh(cov)

            # Descriptive statistics: mean, std dev, kurtosis, skewness
            ev['mean_x'] = np.mean(x)
            ev['mean_y'] = np.mean(y)
            ev['stddev_x'] = np.std(x)
            ev['stddev_y'] = np.std(y)
            ev['kurtosis_x'] = scipy.stats.kurtosis(x)
            ev['kurtosis_y'] = scipy.stats.kurtosis(y)
            ev['skewness_x'] = scipy.stats.skew(x)
            ev['skewness_y'] = scipy.stats.skew(y)

            # means of cluster
            ev['cog_x'] = np.mean(x[mask])
            ev['cog_y'] = np.mean(y[mask])

            # hillas parameter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev['width'], ev['length'] = np.sqrt(eig_vals)
            delta = np.arctan2(eig_vecs[1, 1], eig_vecs[0, 1])
            ev['delta'] = np.rad2deg(delta)

            # higher order weights in cluster coordinates
            cx = np.cos(delta) * x - np.sin(delta) * y
            cy = np.sin(delta) * x + np.cos(delta) * y
            ev['ckurtosis_x'] = scipy.stats.kurtosis(cx)
            ev['ckurtosis_y'] = scipy.stats.kurtosis(cy)
            ev['cskewness_x'] = scipy.stats.skew(cx)
            ev['cskewness_y'] = scipy.stats.skew(cy)

            # number of photons in biggest cluster
            biggest_cluster = np.argmax(np.bincount(
                clustering.labels[clustering.labels != -1]))
            ev['size'] = len(x[clustering.labels == biggest_cluster])

            # number of clusters
            ev['clusters'] = clustering.number

            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

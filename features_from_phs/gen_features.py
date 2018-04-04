import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
import warnings
from fact.instrument import camera_distance_mm_to_deg


def is_simulation_file(input_file):
    reader = ps.EventListReader(input_file)
    event = next(reader)
    return hasattr(event, 'simulation_truth')


def is_simulation_event(event):
    return hasattr(event, 'simulation_truth')


def gen_features(data_file, sim_file=None):

    """
    This generates a certain set of features from photon-stream simulation
    or data files that can be used for further analyses.

    Inputs:
    data_file:          location of input data file as string
    sim_file:           location of input simulations file as string
                        default: corresponding to name of data file

    return:
    pandas data frame with features

    """

    # read in files
    if is_simulation_file(data_file):
        reader = ps.SimulationReader(
          photon_stream_path=data_file,
          mmcs_corsika_path=sim_file)
    else:
        reader = ps.EventListReader(data_file)

    # initialisation of list of dicts containing generated data
    events = list()

    # loop for events
    for event in reader:

        # safe x, y and t components of Photons. shape = (#photons,3)
        xyt = event.photon_stream.point_cloud
        x, y = xyt[:, :2].T
        x = np.rad2deg(x) / camera_distance_mm_to_deg(1)
        y = np.rad2deg(y) / camera_distance_mm_to_deg(1)

        # clustering of events
        clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream)

        # empty dict for values
        ev = {}

        # only calculate when there is at least one cluster (if len(x[mask]) >= 1:)
        if clustering.number >= 1:
            # Simulation truth for energy and direction
            # az_offset_between_magnetic_and_geographic_north = -0.12217305
            # about -7 degrees
            # az_offset_between_corsika_and_ceres
            # = - np.pi + az_offset_between_magnetic_and_geographic_north
            if is_simulation_event(event):
                ev['E_true'] = event.simulation_truth.air_shower.energy
                ev['source_position_az'] = np.rad2deg(
                    event.simulation_truth.air_shower.phi + -0.12217305)
                ev['source_position_zd'] = np.rad2deg(
                    event.simulation_truth.air_shower.theta)
                ev['pointing_position_az'] = event.az + 180
                ev['event'] = event.simulation_truth.event
                ev['reuse'] = event.simulation_truth.reuse
                ev['run'] = event.simulation_truth.run
            else:
                ev['run'] = event.observation_info.run
                ev['event'] = event.observation_info.event
                ev['night'] = event.observation_info.night
                ev['timestamp'] = event.observation_info.time
                ev['pointing_position_az'] = event.az
            ev['pointing_position_zd'] = event.zd

            # biggest cluster:
            biggest_cluster = np.argmax(np.bincount(
                clustering.labels[clustering.labels != -1]
            ))
            mask = clustering.labels == biggest_cluster
            ev['cluster_size_ratio'] = (clustering.labels != -1).sum() / mask.sum()

            ev['n_pixel'] = len(np.unique(np.column_stack([x[mask], y[mask]]), axis=0))

            # covariance and eigenvalues/vectors for later calculations
            cov = np.cov(x[mask], y[mask])
            eig_vals, eig_vecs = np.linalg.eigh(cov)

            # Descriptive statistics: mean, std dev, kurtosis, skewness
            ev['kurtosis_x'] = scipy.stats.kurtosis(x[mask])
            ev['kurtosis_y'] = scipy.stats.kurtosis(y[mask])
            ev['skewness_x'] = scipy.stats.skew(x[mask])
            ev['skewness_y'] = scipy.stats.skew(y[mask])

            # means of cluster
            ev['cog_x'] = np.mean(x[mask])
            ev['cog_y'] = np.mean(y[mask])

            # width, length and delta
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev['width'], ev['length'] = np.sqrt(eig_vals)
                delta = np.arctan(eig_vecs[1, 1] / eig_vecs[0, 1])
            ev['delta'] = delta

            # rotate into main component system
            delta_x = x[mask] - ev['cog_x']
            delta_y = y[mask] - ev['cog_y']
            long = np.cos(delta) * delta_x + np.sin(delta) * delta_y
            trans = - np.sin(delta) * delta_x + np.cos(delta) * delta_y

            # higher order weights in cluster coordinates
            ev['kurtosis_long'] = scipy.stats.kurtosis(long)
            ev['kurtosis_trans'] = scipy.stats.kurtosis(trans)
            ev['skewness_long'] = scipy.stats.skew(long)
            ev['skewness_trans'] = scipy.stats.skew(trans)

            # number of photons in biggest cluster
            ev['size'] = mask.sum()

            # number of clusters
            ev['clusters'] = clustering.number

            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

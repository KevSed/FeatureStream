import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
import warnings
from fact.instrument import camera_distance_mm_to_deg


def is_simulation(input_file):
    reader = ps.EventListReader(input_file)
    event = next(reader)
    return hasattr(event, 'simulation_truth')



def gen_features(data_file, sim_file=None):

    """ This generates a certain set of features from photon-stream simulation
        or data files that can be used for further analyses.

    Inputs:
    data_file:          location of input data file as string
    sim_file:           location of input simulations file as string
                        default: corresponding to name of data file

    return:
    pandas data frame with features

    """

    # read in files
    if is_simulation(data_file):
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

        # events from clusters
        # mask = clustering.labels == 0

        # empty dict for values
        ev = {}

        # only calculate when there is at least one cluster (if len(x[mask]) >= 1:)
        if clustering.number >= 1:
            # Simulation truth for energy and direction
            # az_offset_between_magnetic_and_geographic_north = -0.12217305
            # about -7 degrees
            # az_offset_between_corsika_and_ceres
            # = - np.pi + az_offset_between_magnetic_and_geographic_north
            if is_simulation(data_file):
                ev['E_MC'] = event.simulation_truth.air_shower.energy
                ev['source_position_az'] = np.rad2deg(
                    event.simulation_truth.air_shower.phi + -0.12217305)
                ev['source_position_zd'] = np.rad2deg(
                    event.simulation_truth.air_shower.theta)
                ev['pointing_position_az'] = event.az + 180
            else:
                ev['run'] = event.observation_info.run
                ev['event'] = event.observation_info.event
                ev['night'] = event.observation_info.night
                ev['timestamp'] = event.observation_info.time
                ev['pointing_position_az'] = event.az
            ev['pointing_position_zd'] = event.zd

            # biggest cluster:
            biggest_cluster = np.argmax(np.bincount(
                clustering.labels[clustering.labels != -1]))
            mask = clustering.labels == biggest_cluster

            # covariance and eigenvalues/vectors for later calculations
            cov = np.cov(x[mask], y[mask])
            eig_vals, eig_vecs = np.linalg.eigh(cov)

            # Descriptive statistics: mean, std dev, kurtosis, skewness
            ev['mean_x'] = np.mean(x)
            ev['mean_y'] = np.mean(y)
            ev['stddev_x'] = np.std(x)
            ev['stddev_y'] = np.std(y)
            ev['kurtosis_x'] = scipy.stats.kurtosis(x[mask])
            ev['kurtosis_y'] = scipy.stats.kurtosis(y[mask])
            ev['skewness_x'] = scipy.stats.skew(x[mask])
            ev['skewness_y'] = scipy.stats.skew(y[mask])

            # means of cluster
            ev['cog_x'] = np.mean(x[mask])
            ev['cog_y'] = np.mean(y[mask])

            # hillas parameter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev['width'], ev['length'] = np.sqrt(eig_vals)
            delta = np.arctan2(eig_vecs[1, 1], eig_vecs[0, 1])
            ev['delta'] = delta

            # higher order weights in cluster coordinates
            cx = np.cos(delta) * x[mask] - np.sin(delta) * y[mask]
            cy = np.sin(delta) * x[mask] + np.cos(delta) * y[mask]
            ev['kurtosis_long'] = scipy.stats.kurtosis(cx)
            ev['kurtosis_trans'] = scipy.stats.kurtosis(cy)
            ev['skewness_long'] = scipy.stats.skew(cx)
            ev['skewness_trans'] = scipy.stats.skew(cy)

            # number of photons in biggest cluster
            ev['size'] = len(x[clustering.labels == biggest_cluster])

            # number of clusters
            ev['clusters'] = clustering.number

            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

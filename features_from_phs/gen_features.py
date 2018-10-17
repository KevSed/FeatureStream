import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
import warnings
from fact.instrument import camera_distance_mm_to_deg
from fact.instrument.camera import get_border_pixel_mask

az_offset_between_magnetic_and_geographic_north = -0.12217305


def phs2image(lol, lower=0, upper=7000):
    """
    Delete time slices at beginning and end of the event and return an image for Photon Stream events.

    Inputs:
    lol:   Photon Stream list of photon arrival times per pixel

    Optional:
    lower:    lower limit of range of time slices to return
    upper:    upper limit of range of time slices to return

    return:
    image:          Array with number of photons within time-slice intervall [lower, upper] in every of the 1440 pixels

    """

    image = np.array([
        np.sum((lower <= np.array(l)) & (np.array(l) < upper))
        for l in lol
    ])
    return image


def is_simulation_file(input_file):
    reader = ps.EventListReader(input_file)
    event = next(reader)
    return hasattr(event, 'simulation_truth')


def is_simulation_event(event):
    return hasattr(event, 'simulation_truth')


def safe_observation_info(event):
    """
    Safes meta info of event to dict ev

    Inputs:
    -----------------------------------------
    event:  photon stream event from EventListReader or SimulationReader

    Returns:
    -----------------------------------------
    ev:     dictionary with observation infos
    """

    ev = {}
    # Simulation truth for energy and direction
    # az_offset_between_magnetic_and_geographic_north = -0.12217305
    # about -7 degrees
    # az_offset_between_corsika_and_ceres
    # = - np.pi + az_offset_between_magnetic_and_geographic_north
    if is_simulation_event(event):
        ev['corsika_event_header_total_energy'] = event.simulation_truth.air_shower.energy
        ev['source_position_az'] = np.rad2deg(
            event.simulation_truth.air_shower.phi + az_offset_between_magnetic_and_geographic_north)
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

    return ev


def calc_hillas_features_phs(phs, clustering):
    """
    Safes Hillas features from Photon Stream Cluster to dict ev

    Inputs:
    -----------------------------------------
    phs:            Photon Stream of an event
    clustering:     Photon Stream cluster

    Returns:
    -----------------------------------------
    ev:             dictionary with Hillas features
    """

    ev = {}

    # safe x, y and t components of Photons. shape = (#photons,3)
    xyt = phs.point_cloud
    x, y, t = xyt.T
    x = np.rad2deg(x) / camera_distance_mm_to_deg(1)
    y = np.rad2deg(y) / camera_distance_mm_to_deg(1)

    # biggest cluster:
    biggest_cluster = np.argmax(np.bincount(clustering.labels[clustering.labels != -1]))
    mask = clustering.labels == biggest_cluster

    # all clusters
    # mask = clustering.labels != -1
    ev['cluster_size_ratio'] = (clustering.labels != -1).sum() / mask.sum()

    ev['n_pixel'] = len(np.unique(np.column_stack([x[mask], y[mask]]), axis=0))

    # Leakage
    image = phs2image(phs.list_of_lists)
    cleaned_pix = np.zeros(len(image), dtype=bool)

    border_pix = get_border_pixel_mask()
    k = 0
    cleaned_img = np.zeros(len(image))
    for i in range(len(phs.list_of_lists)):
        for j in range(len(phs.list_of_lists[i])):
            if mask[k]:
                cleaned_pix[i] = True
                cleaned_img[i] += 1
            k += 1

    border_ph = [(border_pix[i] and cleaned_pix[i]) for i in range (1440)]
    ev['leakage'] = cleaned_img[border_ph].sum()/mask.sum()

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

    return ev



def gen_features(data_file, eps, sim_file=None):

    """
    This generates a certain set of features from photon-stream simulation
    or data files that can be used for further analyses.

    Inputs:
    -----------------------------------------
    data_file:          location of input data file as string
    sim_file:           location of input simulations file as string
                        default: corresponding to name of data file

    Returns:
    -----------------------------------------
    pandas data frame with features

    """

   # if eps is None:
   #     eps = 0.1

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
        # clustering of events
        clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream, eps=eps)
        if clustering.number >= 1:
            # empty dict for values
            ev = {}
            # safe hillas features
            ev.update(calc_hillas_features_phs(event.photon_stream, clustering))
            # safe observation info
            ev.update(safe_observation_info(event))
            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

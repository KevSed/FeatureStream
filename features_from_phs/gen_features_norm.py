import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
import warnings
from fact.instrument import camera_distance_mm_to_deg, get_pixel_dataframe
from fact.instrument.camera import get_neighbor_matrix, get_pixel_coords, get_border_pixel_mask
from .gen_features import is_simulation_file, is_simulation_event, safe_observation_info, phs2image


def facttools_cleaning(image, lol, picture_thresh=5.5, boundary_thresh=2):
    """
    Per pixel threshold based cleaning for Photon Stream events.

    Inputs:
    image:          Pixel Image from Photon Stream data
    picture_thresh: Picture threshold for the cleaning
    boundary_thresh: Boundary threshold for the cleaning

    return:
    cleaned_pix:    Boolean array of size 1440 with pixels containing the desired amount of photons

    """


    # matrix containing neighbor information
    neighbor_matrix = get_neighbor_matrix()

    # select pixels above picture_thresh
    # 1. Find pixels containing more photons than an upper threshold
    pix_above_pic = image >= picture_thresh

    # number of neighboring pixels above picture_thresh for each pixel
    number_of_neighbors_above_picture = neighbor_matrix.dot(pix_above_pic.view(np.byte))

    # pixels above picture_thresh and with more than 2 neighboring pixels above same thresh
    # 2. remove pixels with less than 2 neighbors above that threshold
    pix_in_pic = pix_above_pic & (number_of_neighbors_above_picture >= 2)

    # 3. Add neighbors of the remaining pixels that are above boundary_thresh
    pix_above_boundary = (image >= boundary_thresh)
    pix_in_pic_bound = np.zeros(1440, dtype=bool)
    for i in range(1440):
        if pix_in_pic[i]:
            for j in range(1440):
                if pix_above_boundary[j] and neighbor_matrix[i, j]:
                    pix_in_pic_bound[i] = True

    # arrival times per pixel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arrival_times = np.array([np.nanmedian(l) for l in lol])
        arrival_times[np.isnan(arrival_times)] = 0
    # 4. Remove pixels that have less than 2 neighbors with an arrival time inside 5ns
    pix_in_time_intervall = np.zeros(1440, dtype=bool)
    for i in range(1440):
        neighbors = 0
        if pix_in_pic_bound[i]:
            for j in range(1440):
                if pix_in_pic_bound[j] and neighbor_matrix[i, j] and np.abs(arrival_times[i] - arrival_times[j]) <= 10:
                    neighbors += 1
                if neighbors ==2:
                    pix_in_time_intervall[i] = True
                    break

    # number of neighboring pixels
    number_of_neighbors = neighbor_matrix.dot(pix_in_time_intervall.view(np.byte))

    # pixels with more than 2 neighbors
    # 5. Remove single pixels with less than 2 neighbors in the remaining pixels
    pix_with_neighbors = number_of_neighbors >= 2

    # 6. Remove pixels that have less than 2 neighbors with an arrival time inside 5ns
    cleaned_pix = np.zeros(1440, dtype=bool)
    for i in range(1440):
        neighbors = 0
        if pix_with_neighbors[i]:
            for j in range(1440):
                if pix_with_neighbors[j] and neighbor_matrix[i, j] and np.abs(arrival_times[i] - arrival_times[j]) <= 10:
                    neighbors += 1
                if neighbors ==2:
                    cleaned_pix[i] = True
                    break

    return cleaned_pix


def cleaning(image, lol, picture_thresh=5.5, boundary_thresh=2):
    """
    Per pixel threshold based cleaning for Photon Stream events.

    Inputs:
    image:          Pixel Image from Photon Stream data
    picture_thresh: Picture threshold for the cleaning
    boundary_thresh: Boundary threshold for the cleaning

    return:
    cleaned_pix:    Boolean array of size 1440 with pixels containing the desired amount of photons

    """


    # matrix containing neighbor information
    neighbor_matrix = get_neighbor_matrix()

    # select pixels above picture_thresh
    # 1. Find pixels containing more photons than an upper threshold
    pix_above_pic = image >= picture_thresh

    # number of neighboring pixels above picture_thresh for each pixel
    number_of_neighbors_above_picture = neighbor_matrix.dot(pix_above_pic.view(np.byte))

    # pixels above picture_thresh and with more than 2 neighboring pixels above same thresh
    # 2. remove pixels with less than 2 neighbors above that threshold
    pix_in_pic = pix_above_pic & (number_of_neighbors_above_picture >= 2)

    # pixels above boundary thresh
    pix_above_boundary = image >= boundary_thresh
    # pixels with neighbors above picture thresh
    pix_with_pic_neighbors = neighbor_matrix.dot(pix_in_pic)
    # pixels with neighbors above boundary thresh
    pix_with_boundary_neighbors = neighbor_matrix.dot(pix_above_boundary)

    # cleaned pixels  must contain (#photons >= boundary thresh & neighboring pixel above picture thresh) or
    # (#photons >= picture thresh & >= 2 neighbors above picture thresh & at least one neighboring pixel with
    # #photons >= boundary thresh)
    cleaned_pix = ((pix_above_boundary & pix_with_pic_neighbors) | (pix_in_pic & pix_with_boundary_neighbors))

    return cleaned_pix


def calc_hillas_features_image(image, mask):
    """
    Safes hillas features from image to dict ev

    Inputs:
    -----------------------------------------
    image:  Number of photons per pixel (1440)
    mask:   List of pixels that survived the cleaning

    Returns:
    -----------------------------------------
    ev:     dictionary with observation infos
    """

    ev = {}
    x, y = get_pixel_coords()
    ev['n_pixel'] = mask.sum()

    # means of cluster
    ev['cog_x'] = np.average(x[mask], weights=image[mask])
    ev['cog_y'] = np.average(y[mask], weights=image[mask])

    # covariance and eigenvalues/vectors for later calculations
    cov = np.cov(x[mask], y[mask], fweights=image[mask])
    eig_vals, eig_vecs = np.linalg.eigh(cov)

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
    m3_long = np.average(long**3, weights=image[mask])
    m3_trans = np.average(trans**3, weights=image[mask])

    m4_long = np.average(long**4, weights=image[mask])
    m4_trans = np.average(trans**4, weights=image[mask])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev['skewness_long'] = m3_long / ev['length']**3
        ev['skewness_trans'] = m3_trans / ev['width']**3
        ev['kurtosis_long'] = m4_long / ev['length']**4
        ev['kurtosis_trans'] = m4_trans / ev['width']**4

    return ev


def gen_features_norm(data_file, lower, upper, sim_file=None):

    """
    This generates a certain set of features from photon-stream simulation
    or data files that can be used for further analyses.

    Inputs:
    data_file:          location of input data file as string
    lower:              lower limit for time slice cleaning
    upper:              upper limit for time slice cleaning
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
    border_pix = get_border_pixel_mask()
    x, y = get_pixel_coords()

    # loop for events
    for event in reader:

        lol = event.photon_stream.list_of_lists
        image = phs2image(lol, lower, upper)
        mask = cleaning(image, lol)


        # empty dict for values
        ev = {}
        # number of photons in biggest cluster
        ev['size'] = image[mask].sum()


        if ev['size'] > 0:

            border_ph = [(border_pix[i] and mask[i]) for i in range(1440)]
            ev['leakage'] = image[border_ph].sum()/ev['size']
            ev.update(safe_observation_info(event))
            ev.update(calc_hillas_features_image(image, mask))
            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

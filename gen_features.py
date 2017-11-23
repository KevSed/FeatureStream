import photon_stream as ps
import numpy as np
import scipy
import pandas as pd
from fact.io import to_h5py
import sys


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = fill * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# data file
reader = ps.EventListReader("/net/big-tank/POOL/projects/fact/photon-stream/pass4/phs/2013/10/03/20131003_105.phs.jsonl.gz")
events = list()

# loop for events
#for event in reader:
    #print(event)
items = list(range(0, 18017))
le = len(items)
j = 0
for event in reader:
    j = j+1
    # event.photon_stream.list_of_lists: list of photons and their arrival times at certain pixels
    # number of photons
    # image = np.array([len(pixel) for pixel in event.photon_stream.list_of_lists])
    #plot = camera(image)
    #plt.colorbar(plot)

    # safe x, y and t components of Photons. shape = (#photons,3)
    xyt = event.photon_stream.xyt
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
        # clusters = np.bincount(clustering.labels + 1)[1]


        ev = {'cog_x': cog_x, 'cog_y': cog_y, 'mean_x': cmean_x, 'stddev_x': cstd_x, 'stddev_y': cstd_y, 'mean_y': cmean_y, 'width': width, 'length': length, 'angle': angle, 'kurtosis_x': ckurtosis_x, 'kurtosis_y': ckurtosis_y, 'skewness_x': cskewness_x, 'skewness_y': cskewness_y, 'clusters': clusters, 'size': size}
        events.append(ev)
        # w.append(width)
        # l.append(length)
        # a.append(angle)
    print_progress(j + 1, le, prefix='Progress:', suffix='Complete')
# df = pd.DataFrame({'width': w, 'length': l, 'angle': a})
df = pd.DataFrame(events)
to_h5py('features3.hdf5', df, key='events')

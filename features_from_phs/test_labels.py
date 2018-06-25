import numpy as np
import photon_stream as ps
import pandas as pd
from fact.io import to_h5py, read_h5py
from IPython import embed
from tqdm import tqdm


def main():

    reader = ps.EventListReader('/home/ksedlaczek/stream_data/crab/20131101_153.phs.jsonl.gz')

    events = []

    for event in tqdm(reader):
        if len(events) < 1000:
           ev ={}
           clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream)

           ev['labels'] = clustering.labels
           ev['clusters'] = clustering.number
#           ev['biggest_cluster'] = np.argmax(np.bincount(
#               clustering.labels[clustering.labels != -1]
#               ))
#           ev['size'] = len(clustering.labels[clustering.labels == ev['biggest_cluster']])

           events.append(ev)
        else: break

    df = pd.DataFrame(events)
    df.to_csv('test_labels.csv')

    l = pd.read_csv('test_labels.csv')
    embed()



if __name__ == '__main__':
    main()

import photon_stream as ps
import pandas as pd


def is_simulation_file(input_file):
    reader = ps.EventListReader(input_file)
    event = next(reader)
    return hasattr(event, 'simulation_truth')


def is_simulation_event(event):
    return hasattr(event, 'simulation_truth')


def cluster_labels(data_file, sim_file=None):

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
        clustering = ps.photon_cluster.PhotonStreamCluster(event.photon_stream)

        # only calculate when there is at least one cluster (if len(x[mask]) >= 1:)
        if clustering.number >= 1:

            # empty dict for values
            ev = {}

            # Simulation truth for energy and direction
            if is_simulation_event(event):
                ev['event'] = event.simulation_truth.event
                ev['run'] = event.simulation_truth.run
                ev['reuse'] = event.simulation_truth.reuse
            else:
                ev['run'] = event.observation_info.run
                ev['event'] = event.observation_info.event
                ev['night'] = event.observation_info.night

            ev['cluster_labels'] = clustering.labels
            # append values from dict to list of dicts (events)
            events.append(ev)

    # save list of dicts in pandas data frame
    df = pd.DataFrame(events)
    return df

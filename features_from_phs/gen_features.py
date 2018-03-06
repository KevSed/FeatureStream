from .gen_sim_features import gen_sim_features
from .gen_data_features import gen_data_features


def gen_features(file_path):
    try:
        return gen_sim_features(file_path)
    except FileNotFoundError:
        return gen_data_features(file_path)

import photon_stream as ps
from features_from_phs import is_simulation_file

input_file = 'tests/resources/065415.phs.jsonl.gz'

def test():
    assert is_simulation_file(input_file) == True

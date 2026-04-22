import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
__package__ = 'src.models'

from ..data_utils import get_correlations_by_position, load_PL_dataset


if __name__ == "__main__":

    df = load_PL_dataset()

    correlations = get_correlations_by_position(df, positions=['D'], cutoff=0.8)

    print(list(correlations['D'].items()))

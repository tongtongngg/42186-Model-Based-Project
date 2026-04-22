from src.data_utils import get_correlations_by_position, load_PL_dataset


if __name__ == "__main__":

    df = load_PL_dataset()

    correlations = get_correlations_by_position(df, positions=['D'], cutoff=0.6)

    print(list(correlations['M'].items()))

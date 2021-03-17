class MissionIndices:
    @staticmethod
    def get_arche_high_res():
        training_missions = ['8d1b..0000', '25b1..0000', 'ef8b..0000', 'b03a..0000', '0167..0000', '472b..0000', '0282..0000',
                             'e2da..0000', '8a4a..0000', '657d..0000', 'f760..0000', '73cc..0000', '0569..0000', '174e..0000', 'b52f..0000', '298d..0000']
        test_missions = ['89de..0000', '96af..0000', 'd530..0000',
                         'd662..0000', '62d2..0000', '6fec..0000', 'd778..0000']
        return training_missions, test_missions

    @staticmethod
    def get_arche_low_res():
        training_missions = ['7799..0000',
                             '4e53..0000', 'ab54..0000', '6725..0000']
        test_missions = ['2d2b..0000', '6d25..0000']
        return training_missions, test_missions


if __name__ == "__main__":
    from data_source import DataSource
    from database_parser import DatabaseParser
    from training_set import TrainingSet
    import numpy as np

    n_data = 50
    cache = 8
    training_missions, test_missions = MissionIndices.get_arche_low_res()
    print('Loading training data...')
    print(training_missions)
    dataset_path = '/mnt/data/datasets/Spherical/test_training/'
    db_parser = DatabaseParser(dataset_path)
    training_indices, test_indices = db_parser.extract_training_and_test_indices(
        training_missions, test_missions)
    print(f'Found {training_indices.size} training and {test_indices.size} test data')
    idx = np.array(training_indices['idx'].tolist())
    print(idx)

    ds_train = DataSource(dataset_path, cache)
    ds_train.load(n_data, idx)

    ts = TrainingSet(restore=False, bw=100)
    ts.generateAll(ds_train)

    for i in range(0, 10):
        print(f'Processing feature {i}')
        a, p, n = ts[i]
        assert a is not None
        assert p is not None
        assert n is not None

    #ds_test = DataSource(dataset_path, n_data)
    #ds_test.load(n_data, test_indices)

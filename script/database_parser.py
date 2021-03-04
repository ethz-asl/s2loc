import numpy as np
import pandas as pd

from tqdm.auto import tqdm, trange


class DatabaseParser(object):
    def __init__(self, path_to_datasource):
        self.datasource = path_to_datasource
        missions_db_path = path_to_datasource + 'missions.csv'
        self.missions_df = self.read_databases(
            missions_db_path)

    def read_databases(self, missions_db_path):
        print(f'Reading missions db from {missions_db_path}')
        missions_df = pd.read_csv(missions_db_path, names=[
            'mission_anchor', 'mission_positive', 'mission_negative'], delimiter=',', comment='#', header=None)
        print(f'Read {missions_df} in total.')
        print(f'Read {int(missions_df.size/3)} entries.')
        return missions_df

    def extract_training_and_test_indices(self, training_missions, test_missions):
        training_indices = self._extract_train_indices(training_missions, self.missions_df)
        test_indices = self._extract_test_indices(test_missions, self.missions_df)
        if not training_indices.empty and not test_indices.empty:
            natural_join_test_training = test_indices.join(
                training_indices.set_index('idx'), on='idx', how='inner')
            assert(natural_join_test_training.size == 0)
        return training_indices, test_indices

    def _extract_train_indices(self, missions, missions_df):
        indices = pd.DataFrame()
        for i in tqdm(range(0, len(missions))):
            current_df = missions_df[missions_df['mission_anchor']
                                     == missions[i]]
            if current_df.empty:
                continue
            mask = [False] * current_df['mission_positive'].size
            for j in range(0, len(missions)):
                mask = mask | current_df['mission_positive'].str.contains(
                    missions[j]).values

            index_df = pd.DataFrame({'idx': current_df[mask].index})
            indices = indices.append(index_df)

        indices.drop_duplicates()
        return indices

    def _extract_test_indices(self, missions, missions_df):
        indices = pd.DataFrame()
        for i in tqdm(range(0, len(missions))):
            mask = [False] * missions_df['mission_positive'].size
            mask =  mask | missions_df['mission_anchor'].str.contains(missions[i]).values
            mask =  mask | missions_df['mission_positive'].str.contains(missions[i]).values
            current_df = missions_df[mask]
            if current_df.empty:
                continue
            index_df = pd.DataFrame({'idx': current_df.index})
            indices = indices.append(index_df)

        indices.drop_duplicates()
        return indices


if __name__ == "__main__":
    #db_parser = DatabaseParser('/mnt/data/datasets/Spherical/test_training/')
    db_parser = DatabaseParser('/mnt/data/datasets/alice/arche_high_res/')
    #db_parser = DatabaseParser('/tmp/training/')
    training_missions = ['8d1b..0000', '25b1..0000', 'ef8b..0000', 'b03a..0000', '0167..0000', '472b..0000', '0282..0000',
                         'e2da..0000', '8a4a..0000', '657d..0000', 'f760..0000', '73cc..0000', '0569..0000', '174e..0000', 'b52f..0000', '298d..0000']
    test_missions = ['89de..0000', '96af..0000', 'd530..0000',
                     'd662..0000', '62d2..0000', '6fec..0000', 'd778..0000']

    training_indices, test_indices = db_parser.extract_training_and_test_indices(
        training_missions, test_missions)

    print(f'Found {training_indices.size} training and {test_indices.size} test data')
    print(training_indices.head(10))
    print(test_indices.head(10))

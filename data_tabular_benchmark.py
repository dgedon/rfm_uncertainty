import os
import ipdb
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMREG_MED_IDS = {
    44132,  # 0 cpu-act
    44133,  # 1 pol
    44134,  # 2 elevators
    44135,  # 3 isolet
    44136,  # 4 wine-quality
    44137,  # 5 Ailerons
    44138,  # 6 houses
    44139,  # 7 house-16H
    44141,  # 8 Brazilian-houses
    44142,  # 9 Bike-Sharing-Demand
    44144,  # 10 house-sales
    44145,  # 11 sulfur
    44147,  # 12 MiamiHousing2016
    44148,  # 13 superconduct
    44025,  # 14 california
    44026,  # 15 fifa
}


class DataTabularBenchmarkRegression(object):
    def __init__(self, test_ratio=0.3, random_state=42,
                 verbose=False):
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.verbose = verbose

    def return_dataset_names(self):
        dataset_names = []
        for i in range(len(self)):
            _, _, _, _, dataset_name = load_data_tabbenchmark_regression(i)
            dataset_names.append(dataset_name)
        return dataset_names

    def __repr__(self):
        return f"TabularBenchmarkData(idx={self.idx})"

    def __len__(self):
        return len(NUMREG_MED_IDS)

    def get_data(self, dataset_idx: int):
        assert dataset_idx < len(self), f"dataset_idx {dataset_idx} out of range"
        # load the data
        X, y, _, _, dataset_name = load_data_tabbenchmark_regression(dataset_idx)
        # data from pandas to numpy
        X, y = X.values, y.values
        # y to float
        y = y.astype(np.float32)

        # split
        if self.test_ratio == 0:
            X_train, X_test, y_train, y_test = X, [], y, []
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_ratio,
                random_state=self.random_state,
            )

        if self.verbose:
            if self.test_ratio == 0:
                print(f'Loaded dataset {dataset_idx}: {dataset_name} with {X_train.shape[0]} train samples '
                      f'and 0 test samples and {X_train.shape[1]} features')
            else:
                print(f'Loaded dataset {dataset_idx}: {dataset_name} with {X_train.shape[0]} train samples '
                      f'and {X_test.shape[0]} test samples and {X_train.shape[1]} features')

        return (X_train, y_train), (X_test, y_test), dataset_name


def load_data_tabbenchmark_regression(idx=None):
    """
    Load the tabular benchmark data from OpenML
    https://github.com/LeoGrin/tabular-benchmark

    ids from https://arxiv.org/pdf/2207.08815.pdf, appendix A.1.2
    """
    assert idx < len(NUMREG_MED_IDS), f"idx {idx} out of range "
    dataset_id = list(NUMREG_MED_IDS)[idx]
    # download the OpenML task
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                    target=dataset.default_target_attribute
                                                                    )
    return X, y, categorical_indicator, attribute_names, dataset.name


if __name__ == '__main__':
    # test
    data = DataTabularBenchmarkRegression(test_ratio=0.3, random_state=42, verbose=True)
    for i in range(len(data)):
        (X_train, y_train), (X_test, y_test), dataset_name = data.get_data(i)

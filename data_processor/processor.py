from .datasets.separated_svm import get_data as sep_get_data
from .datasets.overlapped_svm import get_data as overl_get_data
from .datasets.weather import get_data as weth_get_data
from .datasets.houseprice import get_data as hp_get_data
from .datasets.cancer import get_data as cn_get_data
from .datasets.ecoli import get_data as ec_get_data


def retrieve_and_process_data(dataset, test_proportion):
    if dataset == "separable_svm":
        return sep_get_data()
    elif dataset == "overlapped_svm":
        return overl_get_data()
    elif dataset == "weather":
        return weth_get_data(test_proportion=test_proportion)
    elif dataset == "cancer":
        return cn_get_data(test_proportion=test_proportion)
    elif dataset == "ecoli":
        return ec_get_data(test_proportion=test_proportion)
    elif dataset == 'houseprice':
        return hp_get_data()
    else:
        raise Exception(f"Unknown dataset:{dataset} passed!!!")

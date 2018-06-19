import pickle
from collections import OrderedDict

def split_list_in_chunks(input_list, num_chunks):
    return [input_list[i::num_chunks] for i in range(num_chunks)]

def load_dict_from_disk(path_to_dict):
    with open(path_to_dict, 'rb') as f:
        return pickle.load(f)

def create_sorted_dict(sorted_list):
    """

    :param sorted_list: Sorted list of tuples (key,value)
    :rtype: dict
    """
    sorted_dict = OrderedDict()

    for tuple in sorted_list:
        word, freq = tuple
        sorted_dict[word] = freq
    return sorted_dict
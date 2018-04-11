from collections import OrderedDict
from utilities.constants import POSITION_VEC_DIM, PART_OF_SPEECH_VEC_DIM


def get_feature_specification_dict(position_vec_dim=None, part_of_speech_vec_dim=None):
    """
    Get dictionaty containing the information about additional features. If the dimension of a feature is specified,
    then this feature will be used.
    :param position_vec_dim: Dimension of position feature embedding
    :param part_of_speech_vec_dim: Dimension of part-of-speech feature embedding
    :rtype: dict
    """
    feature_spec_dict = OrderedDict()

    if position_vec_dim is not None:
        assert (position_vec_dim,type(int))
        feature_spec_dict[POSITION_VEC_DIM] = position_vec_dim

    if part_of_speech_vec_dim is not None:
        assert (part_of_speech_vec_dim,type(int))
        feature_spec_dict[PART_OF_SPEECH_VEC_DIM] = part_of_speech_vec_dim

    return feature_spec_dict



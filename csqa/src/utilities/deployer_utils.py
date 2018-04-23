from collections import OrderedDict


def get_parameters_from_configuration_file(path_to_config):
    """

    :param path_to_config:
    :return:
    """
    config_param_dict = OrderedDict()

    # ------------------Extract parameters------------------
    with open(path_to_config, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(":@value:")
            parameter = parts[0]
            value = parts[1]
            config_param_dict[parameter] = value

    return config_param_dict

import json
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_dialogues_from_json_file(input_path):
    """
    Laod all dialogues contained in JSON file
    :param input_path: Path to JSON file
    :return:
    """
    log.info("Load JSON dictionaires to %s " % (input_path))
    with open(input_path, 'r') as fp:
        data = json.load(fp)
    return data


def get_files_from_direc(path_to_directory):
    """
    Load JSON from directory or subdirectories
    :param path_to_directory: Path to the root directory
    :rtype: list
    """
    files = []
    if os.path.isdir(path_to_directory):
        log.info("Load files from directory %s" % (path_to_directory))
        for file_or_sub_direc in os.listdir(path_to_directory):
            file_or_sub_direc = os.path.join(path_to_directory,file_or_sub_direc)
            if os.path.isfile(file_or_sub_direc) == False:
                files += get_files_from_direc(file_or_sub_direc)
            else:
                files.append(file_or_sub_direc)
    else:
        print(os.listdir(path_to_directory))
        raise Exception("%s is not a directory" % (path_to_directory))

    return files



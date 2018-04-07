import json
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_dialogues_from_json_file(input_path):
    """
    Laod all dialogues contained in JSON file.
    :param input_path: Path to JSON file
    :return:
    """
    log.debug("Load dialogue from %s " % (input_path))
    with open(input_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def get_files_from_direc(path_to_directory):
    """
    Load JSON from directory or subdirectories. Ignores hidden directories.
    :param path_to_directory: Path to the root directory
    :rtype: list
    """

    if Path(path_to_directory).name.startswith('.'):
        raise Exception("Root directory shouldn't be a hidden directory.")

    files = []
    if os.path.isdir(path_to_directory):
        log.debug("Load files from directory %s" % (path_to_directory))
        for file_or_sub_direc in os.listdir(path_to_directory):

            file_or_sub_direc = os.path.join(path_to_directory, file_or_sub_direc)

            # Ignore hidden dicetories
            if Path(file_or_sub_direc).name.startswith('.'):
                continue

            if not os.path.isfile(file_or_sub_direc):
                files += get_files_from_direc(file_or_sub_direc)
            else:
                files.append(file_or_sub_direc)
    else:
        raise Exception("%s is not a directory" % (path_to_directory))

    return files

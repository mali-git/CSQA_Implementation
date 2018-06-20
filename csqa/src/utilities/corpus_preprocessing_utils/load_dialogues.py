import json
import logging
import timeit
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def retrieve_dialogues(input_directory):
    """
    Retreieve dialogues from disk and save in a dict.
    :param input_directory: Path to the root directory
    :rtype: dict
    """
    files = get_files_from_direc(input_directory)
    dialogues_data_dict = _create_dialogue_data_dict(files=files)
    return dialogues_data_dict

def _create_dialogue_data_dict(files):
    log.info("Extract dialogues from files")
    start = timeit.default_timer()
    dialogues_dict = dict()
    for file in files:
        log.debug("Process file %s" % file)
        try:
            dialogue_in_file = load_data_from_json_file(file)
        except:
            log.info("Problem with file %s " % (file))
            continue

        parts_of_path = Path(file).parts
        key = os.path.join(parts_of_path[-3], parts_of_path[-2], parts_of_path[-1])
        dialogues_dict[key] = dialogue_in_file
        log.debug("File %s loaded." % (file))

    stop = timeit.default_timer()
    log.info("Dialogues extracted. It took %s seconds \n" % (str(round(stop - start))))
    return dialogues_dict


def load_data_from_json_file(input_path):
    """
    Laod all dialogues contained in JSON file.
    :param input_path: Path to JSON file
    :rtype: dict
    """
    log.debug("Load data from %s " % (input_path))
    with open(input_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def get_files_from_direc(path_to_directory):
    """
    Load JSON from directory or subdirectories. Ignores hidden directories.
    :param path_to_directory: Path to the root directory
    :rtype: list
    """
    log.info("Load files from %s" % (path_to_directory))
    start = timeit.default_timer()
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

    stop = timeit.default_timer()
    log.info("Files loaded. It took %s seconds \n" % (str(round(stop - start))))
    return files

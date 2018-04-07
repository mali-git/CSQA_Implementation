import logging
import timeit
from pathlib import Path

import click
import os

from utilities.corpus_preprocessing.load_dialogues import get_files_from_direc, load_dialogues_from_json_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_dialogue_data_dict(files):
    dialogues_dict = dict()
    for file in files:
        log.debug("Process file %s" % file)
        try:
            dialogue_in_file = load_dialogues_from_json_file(file)
        except:
            log.info("Problem with file %s " % (file))
            continue

        parts_of_path = Path(file).parts
        key = os.path.join(parts_of_path[-3], parts_of_path[-2], parts_of_path[-1])
        dialogues_dict[key] = dialogue_in_file
        log.debug("File %s loaded." % (file))

    return dialogues_dict


@click.command()
@click.option('-input', help='path to corpus directory', required=True)
def main(input):
    start = timeit.default_timer()
    input_direc = input

    log.info("Load files from %s" % (input_direc))
    files = get_files_from_direc(input_direc)
    stop = timeit.default_timer()
    log.info("Files loaded. It took %s seconds \n" % (str(round(stop - start))))

    # In train directory there are n subdirectories QA_0,...,QA_n
    # Either load all files from all directories or from specific directory. Use -input param
    # For test prupose only specific subdirectory is used
    # Filenames are the keys and the instances in the files are saved as as a list
    log.info("Extract dialogues from files")
    start = timeit.default_timer()
    dialogues_data_dict = create_dialogue_data_dict(files=files)
    stop = timeit.default_timer()
    log.info("Dialogues extracted. It took %s seconds \n" % (str(round(stop - start))))

    for key, dialogue in dialogues_data_dict.items():
        print("User: %s \n " % (dialogue[0]))
        print("System %s" % (dialogue[1]))
        print("------------------------\n \n")


if __name__ == '__main__':
    main()

import logging
import click

from utilities.corpus_preprocessing_utils.load_dialogues import retrieve_dialogues

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@click.command()
@click.option('-input', help='path to corpus directory', required=True)
def main(input):
    input_direc = input

    # In train directory there are n subdirectories QA_0,...,QA_n
    # Either load all files from all directories or from specific directory. Use -input param
    # For test prupose only specific subdirectory is used
    # Filenames are the keys and the instances in the files are saved as as a list
    dialogues_data_dict = retrieve_dialogues(input_directory=input_direc)

    for key, dialogue in dialogues_data_dict.items():
        questions = dialogue[0::2]
        answers = dialogue[1::2]
        for i in range(len(questions)):
            log.info("User: %s \n " % (questions[i]))
            log.info("System %s" % (answers[i]))
            log.info("------------------------\n \n")


if __name__ == '__main__':
    main()

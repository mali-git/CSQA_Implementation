
import click
import logging
from utilities.corpus_preprocessing_utilities.load_dialogues import get_files_from_direc, load_dialogues_from_json_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@click.command()
@click.option('-input', help='path to corpus directory', required=True)
def main(input):
    input_direc = input
    log.info("Load files from %s" % (input_direc))
    files = get_files_from_direc(input_direc)

    for file in files:
        instances = load_dialogues_from_json_file(file)
        print(instances)
        for instance in instances:
            print(instance)
        exit(0)



if __name__ == '__main__':
    main()
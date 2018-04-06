import logging
import click
from utilities.corpus_preprocessing.load_dialogues import get_files_from_direc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@click.command()
@click.option('-input', help='path to corpus directory', required=True)
def main(input):
    input_direc = input
    log.info("Load files from %s" % (input_direc))
    files = get_files_from_direc(input_direc)


if __name__ == '__main__':
    main()

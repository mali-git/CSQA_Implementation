import logging
from collections import OrderedDict
import timeit
from multiprocessing import Pool
import multiprocessing
import spacy

from utilities.constants import CSQA_UTTERANCE
from utilities.corpus_preprocessing_utils.load_dialogues import retrieve_dialogues
from utilities.general_utils import split_list_in_chunks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
nlp_parser = spacy.load('en')

def update_word_freqs(utter_txt):
    max_freq = 100000
    word_freq_dict = OrderedDict()
    doc = nlp_parser(u'%s' % (utter_txt))

    for token in doc:
        token = token.lower_
        if token in word_freq_dict:
            word_freq = word_freq_dict[token]
            if word_freq < max_freq:
                word_freq_dict[token] = word_freq + 1
        else:
            word_freq_dict[token] = 1

    return word_freq_dict

def create_vocab(chunck_dialogues):
    """

    :param chunck_dialogues:
    :return:
    """
    context_word_freqs = OrderedDict()
    respose_word_freqs = OrderedDict()
    max_freq = 100000

    for dialogue in chunck_dialogues:
        context_utters = dialogue[:-1]
        resonse = dialogue[-1]

        for utter_dict in context_utters:
            utterance_txt = utter_dict[CSQA_UTTERANCE]
            context_word_freqs.update(update_word_freqs(utter_txt=utterance_txt))

        response_txt = resonse[CSQA_UTTERANCE]
        respose_word_freqs.update(update_word_freqs(utter_txt=response_txt))

    return context_word_freqs, respose_word_freqs




def create_csqa_vocabs(input_direc):
    """
    Create vocabulary for CSQA dataset. Contains word ad their frequency in the dataset
    :param input_direc: Path to input directory
    :rtype: dict
    """

    dialogues_data_dict = retrieve_dialogues(input_directory=input_direc)
    num_processes = multiprocessing.cpu_count()
    start = timeit.default_timer()

    dialogues = list(dialogues_data_dict.values())
    chunks = split_list_in_chunks(input_list=dialogues, num_chunks=num_processes)
    tuples = None

    with Pool(num_processes) as p:
        tuples = p.map(create_vocab, chunks)

    stop = timeit.default_timer()
    log.info("Computation of sum took %s seconds \n" % (str(round(stop - start))))

    return tuples


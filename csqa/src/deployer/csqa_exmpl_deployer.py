from collections import OrderedDict
import logging
import click
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from neural_network_models.csqa_network import CSQANetwork
from utilities.constants import EMBEDDED_SEQUENCES, EMBEDDED_RESPONSES, NUM_UNITS_HRE_UTTERANCE_CELL, \
    NUM_UNITS_HRE_CONTEXT_CELL, NUM_HOPS, WORD_VEC_DIM, VOCABUALRY_SIZE, LEARNING_RATE, OPTIMIZER, ADAM, \
    MAX_NUM_UTTER_TOKENS


@click.command()
@click.option('-vocab_path', help='path to vocabulary file', required=True)
@click.option('-keys_path', help='path to file containing embedded keys', required=True)
@click.option('-values_path', help='path to file containing embedded values', required=True)
def main(vocab_path, keys_path, values_path):
    utter_1 = 'We are building a new dialogue system'
    utter_2 = 'What is a dialogue system?'
    keys = ['rel_1_subj_1', 'rel_1_subj_2', 'rel_1_subj_3']
    values = ['obj_1', 'obj_2', 'obj_3']
    num_trainable_tokens = 3
    max_length = 10

    voc_to_id, id_to_vocab, voc_embeddings = process_embedding_file(path_to_file=vocab_path)
    keys_to_id, id_to_keys, key_embeddings = process_embedding_file(path_to_file=keys_path)
    values_to_id, id_to_values, value_embeddings = process_embedding_file(path_to_file=values_path)

    utter_1_ids = [voc_to_id[word] if word in voc_to_id else voc_to_id['<unk>'] for word in utter_1.split()]
    utter_2_ids = [voc_to_id[word] if word in voc_to_id else voc_to_id['<unk>'] for word in utter_2.split()]

    utter_1_ids = add_padding(sequence=utter_1_ids, max_length=max_length, voc_to_id=voc_to_id)
    utter_2_ids = add_padding(sequence=utter_2_ids, max_length=max_length, voc_to_id=voc_to_id)

    batch_keys_embeddings = np.array([key_embeddings.T], dtype=np.float32)
    batch_value_embeddings = np.array([value_embeddings.T], dtype=np.float32)

    instance_1 = OrderedDict()
    instance_1[EMBEDDED_SEQUENCES] = np.array([utter_1_ids, utter_2_ids])
    instance_1[EMBEDDED_RESPONSES] = np.array([utter_2])

    features = OrderedDict()
    features['BATCH_DIALOGUES'] = [instance_1]
    features['BATCH_KEYS'] = batch_keys_embeddings
    features['BATCH VALUES'] = batch_value_embeddings

    log.info('-------BATCH_DIALOGUES-------')
    log.info(len(features['BATCH_DIALOGUES']))

    log.info('-------BATCH_KEYS-------')
    log.info(features['BATCH_KEYS'].shape)

    log.info('-------BATCH VALUES-------')
    log.info(features['BATCH VALUES'].shape)

    # Define mode parameters
    model_params = OrderedDict()
    model_params[NUM_UNITS_HRE_UTTERANCE_CELL] = 10
    model_params[NUM_UNITS_HRE_CONTEXT_CELL] = 5
    model_params[NUM_HOPS] = 2
    model_params[WORD_VEC_DIM] = 2
    model_params[VOCABUALRY_SIZE] = 17
    model_params[LEARNING_RATE] = 0.001
    model_params[OPTIMIZER] = ADAM
    model_params[MAX_NUM_UTTER_TOKENS] = max_length

    # TODO: Adapt stategy for passing arguments
    model = CSQANetwork(num_trainable_tokens=num_trainable_tokens, word_vec_dim=2, vocab_size=17,
                initial_embeddings=voc_embeddings)

    



def process_embedding_file(path_to_file):
    word_to_id = OrderedDict()
    id_to_word = OrderedDict()
    embeddings = []

    with open(path_to_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            parts = line.split()
            word = parts[0].rstrip()
            word_to_id[word] = i
            id_to_word[i] = word
            embedding = [float(emb) for emb in parts[1:]]
            embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    return word_to_id, id_to_word, embeddings


def add_padding(sequence, max_length, voc_to_id):
    padding_size = max_length - len(sequence)
    padding = [voc_to_id['<pad>'] for i in range(padding_size)]

    return sequence + padding


if __name__ == '__main__':
    main()

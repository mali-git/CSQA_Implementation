import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from neural_network_models.csqa_network import CSQANetwork
from utilities.constants import NUM_UNITS_HRE_UTTERANCE_CELL, \
    NUM_UNITS_HRE_CONTEXT_CELL, NUM_HOPS, WORD_VEC_DIM, VOCABUALRY_SIZE, LEARNING_RATE, OPTIMIZER, ADAM, \
    MAX_NUM_UTTER_TOKENS, BATCH_SIZE, NUM_TRAINABLE_TOKENS, RESPONSES


@click.command()
@click.option('-vocab_path', help='path to vocabulary file', required=True)
@click.option('-keys_path', help='path to file containing embedded keys', required=True)
@click.option('-values_path', help='path to file containing embedded values', required=True)
@click.option('-output_direc', help='path to output directory', required=True)
def main(vocab_path, keys_path, values_path, output_direc):
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

    keys_embeddings = np.array([key_embeddings.T], dtype=np.float32)
    value_embeddings = np.array([value_embeddings.T], dtype=np.float32)

    dialogues = np.array([[utter_1_ids, utter_2_ids]], dtype=np.int32)

    log.info('-------DIALOGUES-------')
    log.info(dialogues.shape)

    log.info('-------KEYS-------')
    log.info(keys_embeddings.shape)

    log.info('-------VALUES-------')
    log.info(value_embeddings.shape)

    # Define model parameters
    model_params = OrderedDict()
    model_params[NUM_UNITS_HRE_UTTERANCE_CELL] = 15
    model_params[NUM_UNITS_HRE_CONTEXT_CELL] = 5
    model_params[NUM_HOPS] = 2
    model_params[WORD_VEC_DIM] = 2
    model_params[VOCABUALRY_SIZE] = 17
    model_params[LEARNING_RATE] = 0.001
    model_params[OPTIMIZER] = ADAM
    model_params[MAX_NUM_UTTER_TOKENS] = max_length
    model_params[NUM_TRAINABLE_TOKENS] = num_trainable_tokens
    model_params[BATCH_SIZE] = 1

    currentTime = time.strftime("%H:%M:%S")
    currentDate = time.strftime("%d/%m/%Y").replace('/', '-')
    output_direc += '/' + currentDate + '_' + currentTime + ''
    os.makedirs(output_direc)

    # TODO: Adapt stategy for passing arguments
    model = CSQANetwork(initial_embeddings=voc_embeddings)

    # configuration = tf.contrib.learn.RunConfig(save_checkpoints_secs=3000, gpu_memory_fraction=0.9

    nn = tf.estimator.Estimator(model_fn=model.model_fct, params=model_params,
                                model_dir=output_direc,
                                config=None)

    nn.train(input_fn=lambda: input_fct(dialogues=dialogues, responses=np.array([utter_2_ids]),
                                        keys_embedded=keys_embeddings, values_embedded=value_embeddings,
                                        batch_size=1),
             steps=100)


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


def input_fct(dialogues, responses, keys_embedded, values_embedded, batch_size):
    dialogues, responses, keys_embedded, values_embedded = tf.train.slice_input_producer(
        [dialogues, responses, keys_embedded, values_embedded],
        shuffle=False)

    dataset_dict = dict(dialogues=dialogues, responses=responses, keys_embedded=keys_embedded,
                        values_embedded=values_embedded)

    batch_dicts = tf.train.batch(dataset_dict, batch_size=batch_size,
                                 num_threads=1, capacity=batch_size * 2,
                                 enqueue_many=False, shapes=None, dynamic_pad=False,
                                 allow_smaller_final_batch=False,
                                 shared_name=None, name=None)

    responses = batch_dicts.pop(RESPONSES)

    return batch_dicts, responses


if __name__ == '__main__':
    main()

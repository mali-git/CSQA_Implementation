import logging

import numpy as np
import spacy
from collections import OrderedDict
from gensim.models import KeyedVectors

from utilities.constants import WORD_VEC_DIM, CSQA_UTTERANCE, CSQA_ENTITIES_IN_UTTERANCE, POSITION_VEC_DIM, \
    PART_OF_SPEECH_VEC_DIM
from utilities.corpus_preprocessing.load_dialogues import load_data_from_json_file
from utilities.corpus_preprocessing.text_manipulation_utils import save_insertion_of_offsets, mark_parts_in_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Utterance2TensorCreator(object):
    def __init__(self, max_num_utter_tokens, features_spec_dict, path_to_entity_id_to_label_mapping,
                 path_to_predicate_id_to_label_mapping, word_to_vec_dict=None, path_to_kb_embeddings=None):
        """
        :param max_num_utter_tokens: Maximum length (in tokens) of an utterance
        :param features_spec_dict: dictionary describing which features to use and their dimension
        :param path_to_entity_id_to_label_mapping: Path to file containing the mappings of entity ids to labels
        :param path_to_predicate_id_to_label_mapping: Path to file containing the mappings of predicate ids to labels
        :param word_to_vec_dict: Dictionary containing as keys the paths to the word2Vec models. Values indicate
        whether a model is in binary format or not.
        :param path_to_kb_embeddings: Path to KB embeddings
        """

        assert (word_to_vec_dict is not None)

        if WORD_VEC_DIM not in self.features_spec_dict:
            raise Exception("Specify 'WORD_VEC_DIM'")

        if POSITION_VEC_DIM in self.features_spec_dict:
            log.info("Position embeddings are not supported in current version, but will be available in version 0.1.2")

        self.word_to_vec_models = self.load_word_2_vec_models(word_to_vec_dict=word_to_vec_dict)
        self.out_of_vocab_words_mappings = [OrderedDict() for _ in range(len(self.word_to_vec_models))]
        self.entity_id_to_label_dict = self.load_entity_to_label_mapping(path_to_entity_id_to_label_mapping)
        self.predicate_id_to_label_dict = self.load_entity_to_predicate_mapping(path_to_predicate_id_to_label_mapping)
        self.kg_embeddings_dict = self.load_kg_embeddings(path_to_kb_embeddings=path_to_kb_embeddings)
        self.max_num_utter_tokens = max_num_utter_tokens
        self.features_spec_dict = features_spec_dict
        self.nlp_parser = spacy.load('en')

    def load_word_2_vec_models(self, word_to_vec_dict):
        """
        Loads word2Vec models from disk.
        :param word_to_vec_dict: Dictionary containing as keys the paths to the word2Vec models. Values indicate
        wheter a model is in binary format or not.
        :rtype: list
        """
        word_to_vec_models = []

        for current_path, format in word_to_vec_dict.items():
            word_to_vec_models.append(KeyedVectors.load_word2vec_format(current_path, binary=format))

        return word_to_vec_models

    def load_kg_embeddings(self, path_to_kb_embeddings):
        pass

    def load_entity_to_label_mapping(self, path_to_entity_id_to_label_mapping):
        return load_data_from_json_file(path_to_entity_id_to_label_mapping)

    def load_entity_to_predicate_mapping(self, path_to_predicate_id_to_label_mapping):
        return load_data_from_json_file(path_to_predicate_id_to_label_mapping)

    def create_training_instances(self, dialogue, file_id):
        """
        Computes training instances for a specific dialogue. Call function for each dialogue containd in training set.
        :param dialogue: List containing all utterances of a specific dialogue
        :param file_id: 'directory/filename'. Is needed for mapping.
        :rtype: list
        """
        questions = dialogue[0::2]
        answers = dialogue[1::2]
        assert (len(questions) == len(answers))

        training_instance_dicts = []

        for i in range(len(questions)):
            question = questions[i]
            answer = answers[i]

            # Step 1: Get offsets of utterance-parts based on mentioned entites
            relevant_entity_start_offsets_in_question, relevant_entity_end_offsets_in_question = \
                self.get_offsets_of_relevant_entities_in_utterance(
                    question)
            relevant_entity_start_offsets_in_answer, relevant_entity_end_offsets_in_answer = \
                self.get_offsets_of_relevant_entities_in_utterance(answer)

            question_offset_info_dict = mark_parts_in_text(
                start_offsets_entities=relevant_entity_start_offsets_in_question,
                end_offsets_entities=relevant_entity_end_offsets_in_question,
                text=question[CSQA_UTTERANCE])

            answer_offset_info_dict = mark_parts_in_text(start_offsets_entities=relevant_entity_start_offsets_in_answer,
                                                         end_offsets_entities=relevant_entity_end_offsets_in_answer,
                                                         text=answer[CSQA_UTTERANCE])

            # Step 2: Compute tensor embedding for utterance
            embedded_question = self.compute_tensor_embedding(utterance_dict=question,
                                                              utterance_offsets_info_dict=question_offset_info_dict)
            embedded_answer = self.compute_tensor_embedding(utterance_dict=answer,
                                                            utterance_offsets_info_dict=answer_offset_info_dict)

            # Step 3: Create training instance
            training_instance_dict = self.create_instance_dict(is_training_instance=True)
            training_instance_dicts.append(training_instance_dict)

        return training_instance_dict

    def get_offsets_of_relevant_entities_in_utterance(self, utterance_dict):
        """
        Returns sorted (increasing) offsets of utterance-parts based on mentioned entities
        :param utterance_dict: Dictionary containing all information about passed utterance
        :rtype: dict
        """
        utterance = utterance_dict[CSQA_UTTERANCE]
        ids_of_entities_in_utterance = utterance_dict[CSQA_ENTITIES_IN_UTTERANCE]
        entities_in_utterance = [self.entity_id_to_label_dict[entity_id] for entity_id in ids_of_entities_in_utterance]
        start_offsets, end_offsets = [], []

        for entity_in_utterance in entities_in_utterance:
            # Get offsets of entity
            start = utterance.find(entity_in_utterance)

            if start == -1:
                # Entity not found
                log.info("Entity %s not found in utterance %s" % (entity_in_utterance, utterance))
                continue

            end = start + len(entity_in_utterance)

            # Get updated list of ofsets. Function handles overlaps.
            start_offsets, end_offsets = save_insertion_of_offsets(start_offsets, end_offsets,
                                                                   start, end)

        return start_offsets, end_offsets

    def compute_tensor_embedding(self, utterance_dict, utterance_offsets_info_dict):
        utterance = utterance_dict[CSQA_UTTERANCE]

        for offset_tuple, is_entity in utterance_offsets_info_dict.items():
            embedded_sequence = self.compute_sequence_embedding(text=utterance, offset_tuple=offset_tuple,
                                                                is_entity=is_entity)

    def create_instances_for_prediction(self):
        pass

    def create_instance_dict(self, is_training_instance):
        pass

    def compute_sequence_embedding(self, text, offset_tuple, is_entity):
        """
        Computes the embeddings for a sequence. For each part of the sequence the corresponding embedding is computed.
        :param text: Sequence to embed
        :param offset_tuple: Tuple containing start and position of sequence in text
        :param is_entity: Flag indicating, whether sequence represent and entity.
        :rtype: list
        """
        start = offset_tuple[0]
        end = offset_tuple[1]
        # Additional features
        use_part_of_speech_embedding = False

        if PART_OF_SPEECH_VEC_DIM in self.features_spec_dict:
            use_part_of_speech_embedding = True

        if is_entity:
            entity = text[start:end]
            seq_embedding = self.get_sequence_embedding_for_entity(entity=entity, use_part_of_speech_embedding=
            use_part_of_speech_embedding)
        else:
            # Remove preceding and succeeding whitespaces
            seq = text[start:end].strip()
            # Tokenize
            tokens = [token for token in self.nlp_parser(seq)]
            seq_embedding = [self.get_word_embedding(token) for token in tokens]

            # TODO: Merge word embeddings and part-of-speech embeddings
            if use_part_of_speech_embedding:
                part_of_speech_embeddings = [self.get_part_of_speech_embedding(token=token) for token in tokens]

        return seq_embedding

    def get_sequence_embedding_for_entity(self, entity, use_part_of_speech_embedding=False):
        kg_entity_embedding = self.get_kg_embedding(entity=entity)
        seq_embedding = kg_entity_embedding

        if use_part_of_speech_embedding:
            # TODO: Assign predefined POS tag for entity
            part_of_speech_embedding = self.get_part_of_speech_embedding(token=entity, is_entity=True)
            seq_embedding += part_of_speech_embedding

        return seq_embedding

    def get_kg_embedding(self, entity):
        # TODO: If using severel word2Vec models, concatenate KG embedding #word2Vec models-times
        pass

    def get_word_embedding(self, token):
        embeddigs_of_word = []

        for i, word_to_vec_model in enumerate(self.word_to_vec_models):
            if token in word_to_vec_model:
                embeddigs_of_word += word_to_vec_model[token]
            else:
                out_of_vocab_embedding = self.get_out_of_vocab_embedding(token=token, word_to_vec_model_id=i)
                embeddigs_of_word += out_of_vocab_embedding

        return embeddigs_of_word

    def get_part_of_speech_embedding(self, token, is_entity=False):
        # TODO: If using severel word2Vec models, concatenate POS-tag embedding #word2Vec models-times
        pass

    def get_out_of_vocab_embedding(self, token, word_to_vec_model_id):
        out_of_vocab_words_mapping = self.out_of_vocab_words_mappings[word_to_vec_model_id]

        if token in out_of_vocab_words_mapping:
            return out_of_vocab_words_mapping[token]
        else:
            # Randomly initialize word embedding
            out_of_vocab_embedding = np.random.uniform(low=-0.1, high=0.1,
                                                       size=(self.features_spec_dict[WORD_VEC_DIM],)).tolist()
            out_of_vocab_words_mapping[token] = out_of_vocab_embedding
            return out_of_vocab_embedding

import logging
from collections import OrderedDict

import numpy as np
import spacy
from gensim.models import KeyedVectors

from utilities.constants import WORD_VEC_DIM, CSQA_UTTERANCE, CSQA_ENTITIES_IN_UTTERANCE, POSITION_VEC_DIM, \
    PART_OF_SPEECH_VEC_DIM, INSTANCE_ID, QUESTION_ENTITIES, ANSWER_ENTITIES, QUESTION_UTTERANCE, ANSWER_UTTERANCE, \
    CSQA_RELATIONS, PREDICATE_IDS_QUESTION, QUESTION_UTTERANCE_EMBEDDED, PREDICATE_IDS_ANSWER, ANSWER_UTTERANCE_EMBEDDED
from utilities.corpus_preprocessing.load_dialogues import load_data_from_json_file
from utilities.corpus_preprocessing.text_manipulation_utils import save_insertion_of_offsets, mark_parts_in_text, \
    compute_nlp_features

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Utterance2TensorCreator(object):
    def __init__(self, max_num_utter_tokens, features_spec_dict, path_to_entity_id_to_label_mapping,
                 word_to_vec_dict=None, path_to_kb_embeddings=None):
        """
        :param max_num_utter_tokens: Maximum length (in tokens) of an utterance
        :param features_spec_dict: dictionary describing which features to use and their dimension
        :param path_to_entity_id_to_label_mapping: Path to file containing the mappings of entity ids to labels
        :param word_to_vec_dict: Dictionary containing as keys the paths to the word2Vec models. Values indicate
        whether a model is in binary format or not.
        :param path_to_kb_embeddings: Path to KB embeddings
        """

        assert (word_to_vec_dict is not None)

        if WORD_VEC_DIM not in features_spec_dict:
            raise Exception("Specify 'WORD_VEC_DIM'")

        if POSITION_VEC_DIM in features_spec_dict:
            log.info("Position embeddings are not supported in current version, but will be available in version 0.1.2")

        self.word_to_vec_models = self.load_word_2_vec_models(word_to_vec_dict=word_to_vec_dict)
        self.out_of_vocab_words_mappings = [OrderedDict() for _ in range(len(self.word_to_vec_models))]
        self.entity_id_to_label_dict = self.load_entity_to_label_mapping(path_to_entity_id_to_label_mapping)
        self.kg_embeddings_dict = self.load_kg_embeddings(path_to_kb_embeddings=path_to_kb_embeddings)
        self.max_num_utter_tokens = max_num_utter_tokens
        self.features_spec_dict = features_spec_dict
        self.nlp_parser = spacy.load('en')
        self.part_of_speech_embedding_dict = dict()
        self.dummy_entity_embedding = np.random.uniform(low=-0.1, high=0.1,
                                                        size=(self.features_spec_dict[WORD_VEC_DIM],)).tolist()

    def load_word_2_vec_models(self, word_to_vec_dict):
        """
        Loads word2Vec models from disk.
        :param word_to_vec_dict: Dictionary containing as keys the paths to the word2Vec models. Each value in dict
        is a list where list[0] indicates whether model is in c format and list[1] indicates whether model is saved
        as binary or not
        :rtype: list
        """
        word_to_vec_models = []

        for current_path, format_info in word_to_vec_dict.items():
            is_c_format = format_info[0]
            is_binary = format_info[1]

            if is_c_format:
                word_to_vec_models.append(KeyedVectors.load_word2vec_format(current_path, binary=is_binary))
            else:
                word_to_vec_models.append(KeyedVectors.load(fname_or_handle=current_path))

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
            question_txt = question[CSQA_UTTERANCE]
            answer_txt = answer[CSQA_UTTERANCE]

            # Step 1: Get offsets of utterance-parts based on mentioned entites
            relevant_entity_start_offsets_in_question, relevant_entity_end_offsets_in_question = \
                self.get_offsets_of_relevant_entities_in_utterance(
                    question)
            relevant_entity_start_offsets_in_answer, relevant_entity_end_offsets_in_answer = \
                self.get_offsets_of_relevant_entities_in_utterance(answer)

            question_offset_info_dict = mark_parts_in_text(
                start_offsets_entities=relevant_entity_start_offsets_in_question,
                end_offsets_entities=relevant_entity_end_offsets_in_question,
                text=question_txt)

            answer_offset_info_dict = mark_parts_in_text(start_offsets_entities=relevant_entity_start_offsets_in_answer,
                                                         end_offsets_entities=relevant_entity_end_offsets_in_answer,
                                                         text=answer_txt)

            # Step 2: Compute NLP features
            question_nlp_spans = compute_nlp_features(txt=question_txt,
                                                      offsets_info_dict=question_offset_info_dict)
            answer_nlp_spans = compute_nlp_features(txt=answer_txt,
                                                    offsets_info_dict=answer_offset_info_dict)

            # Step 3: Compute tensor embedding for utterance
            embedded_question = self.compute_tensor_embedding(utterance_dict=question,
                                                              utterance_offsets_info_dict=question_offset_info_dict,
                                                              nlp_spans=question_nlp_spans)
            embedded_answer = self.compute_tensor_embedding(utterance_dict=answer,
                                                            utterance_offsets_info_dict=answer_offset_info_dict,
                                                            nlp_spans=answer_nlp_spans)

            # Step 3: Create training instance
            instance_id = file_id + '_question_' + str(i)

            training_instance_dict = self.create_instance_dict(instance_id=instance_id,
                                                               is_training_instance=True,
                                                               embedded_utterance_one=embedded_question,
                                                               utterance_one_dict=question,
                                                               embedded_utterance_two=embedded_answer,
                                                               utterance_two_dict=answer)
            training_instance_dicts.append(training_instance_dict)

        return training_instance_dict

    def create_instances_for_prediction(self, dialogue, file_id):

        prediction_instance_dicts = []

        for i,utterance in enumerate(dialogue):
            utterance_dict = utterance

            utterance_txt = utterance_dict[CSQA_UTTERANCE]

            # Step 1: Get offsets of utterance-parts based on mentioned entites
            relevant_entity_start_offsets_in_utterance, relevant_entity_end_offsets_in_utterance = \
                self.get_offsets_of_relevant_entities_in_utterance(
                    utterance_dict)

            utterance_offset_info_dict = mark_parts_in_text(
                start_offsets_entities=relevant_entity_start_offsets_in_utterance,
                end_offsets_entities=relevant_entity_end_offsets_in_utterance,
                text=utterance_txt)

            # Step 2: Compute NLP features
            utterance_nlp_spans = compute_nlp_features(txt=utterance_txt,
                                                       offsets_info_dict=utterance_offset_info_dict)

            # Step 3: Compute tensor embedding for utterance
            embedded_utterance = self.compute_tensor_embedding(utterance_dict=utterance_dict,
                                                               utterance_offsets_info_dict=utterance_offset_info_dict,
                                                               nlp_spans=utterance_nlp_spans)

            # Step 3: Create training instance
            instance_id = file_id + '_utterance_' + str(i)

            prediction_instance_dict = self.create_instance_dict(instance_id=instance_id,
                                                                 is_training_instance=True,
                                                                 embedded_utterance_one=embedded_utterance,
                                                                 utterance_one_dict=utterance_dict)
            prediction_instance_dicts.append(prediction_instance_dict)

        return prediction_instance_dict

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

    def compute_tensor_embedding(self, utterance_dict, utterance_offsets_info_dict, nlp_spans):
        utterance = utterance_dict[CSQA_UTTERANCE]
        counter = 0
        embedded_seqs = []

        for offset_tuple, is_entity in utterance_offsets_info_dict.items():
            # [  [ [token-1_model-1 embedding],...,[token-1_model-n embedding] ],...,
            # [ [token-k_model-1 embedding],...,[token-k_model-n embedding] ]  ]

            nlp_span = nlp_spans[counter]
            embedded_seq = self.compute_sequence_embedding(txt=utterance, offset_tuple=offset_tuple,
                                                           is_entity=is_entity, nlp_span=nlp_span)
            embedded_seqs += embedded_seq

        padded_seqs = self.add_padding_to_embedding(seq_embedding=embedded_seqs)
        tensor_embedding = self.create_tensor(embedded_sequence=padded_seqs)

        return tensor_embedding

    def create_instance_dict(self, instance_id, is_training_instance, embedded_utterance_one, utterance_one_dict,
                             embedded_utterance_two=None,
                             utterance_two_dict=None):
        """

        :param embedded_utterance_one:
        :param embedded_utterance_two:
        :param utterance_one_dict:
        :param utterance_two_dict:
        :param instance_id:
        :param is_training_instance:
        :return:
        """

        instance_dict = OrderedDict()
        instance_dict[INSTANCE_ID] = instance_id
        instance_dict[QUESTION_ENTITIES] = utterance_one_dict[CSQA_ENTITIES_IN_UTTERANCE]
        instance_dict[PREDICATE_IDS_QUESTION] = utterance_one_dict[CSQA_RELATIONS]
        instance_dict[QUESTION_UTTERANCE] = utterance_one_dict[CSQA_UTTERANCE]
        instance_dict[QUESTION_UTTERANCE_EMBEDDED] = embedded_utterance_one

        if is_training_instance:
            instance_dict[ANSWER_ENTITIES] = utterance_two_dict[CSQA_ENTITIES_IN_UTTERANCE]
            instance_dict[PREDICATE_IDS_ANSWER] = utterance_two_dict[CSQA_RELATIONS]
            instance_dict[ANSWER_UTTERANCE] = utterance_two_dict[CSQA_UTTERANCE]
            instance_dict[ANSWER_UTTERANCE_EMBEDDED] = embedded_utterance_two

        return instance_dict

    def get_sequence_embedding_for_entity(self, entity, nlp_span, use_part_of_speech_embedding=False):
        """
        Compute the embedding of an entity (single token or several tokens). If several word2Vec models are specified,
        then the entity embedding is returned # word2Vec models-times
        :param entity: Entity for which embedding should be computed
        :param nlp_span: Container containing all NLP features (tokens,POS-tag,dependency parsing etc.) for entity
        :param use_part_of_speech_embedding: Flag indicating whether POS-tag embedding should be computed
        :rtype: list
        """
        kg_entity_embedding = self.get_kg_embedding(entity=entity)
        # Copy entity embedding if several word2Vec models should be used
        seq_embedding = np.repeat(a=[kg_entity_embedding], repeats=len(self.word_to_vec_models), axis=0)

        if use_part_of_speech_embedding:
            # Current strategy for POS-tags of entities: If entity consists of mutilple tokens such as
            # Chancellor of Germany, combine POS-tags for each token --> tag_1-tag_2-_tag-3
            pos_tags = [token.pos_ for token in nlp_span]
            if len(pos_tags) >= 1:
                pos_tag = "-".join(pos_tags)
            else:
                pos_tag = pos_tags[0]

            # Copy POS-tag embedding if several word2Vec models should be used
            part_of_speech_embedding = self.get_part_of_speech_embedding(pos_tag=pos_tag)
            part_of_speech_embeddings = np.repeat(a=[part_of_speech_embedding], repeats=len(self.word_to_vec_models),
                                                  axis=0)

            # Concatenate entity embeddings with POS-Tag embeddings:
            # [ [kg_embedding feature embedding],...,[kg_embedding  feature embedding] ]
            seq_embedding = np.concatenate([seq_embedding, part_of_speech_embeddings], axis=0)

        seq_embedding = [seq_embedding]

        return seq_embedding

    def get_kg_embedding(self, entity):
        # Dummy representation until KG embeddings are available
        #  TODO: Replace implementation
        return self.dummy_entity_embedding

    def get_embeddings_for_token(self, token):
        """
        Extract from each word2Vec model the embedding for the 'token'.
        :param token: The token for wich the word embeddings should be extraced
        :rtype: list: A list of lists. Each list contains the token embedding for a specific word2Vec model:
        [[token embedding based on model-1], ..., [token embedding based on model-n]]
        """
        embeddigs_of_word = []

        for i, word_to_vec_model in enumerate(self.word_to_vec_models):
            if token in word_to_vec_model:
                embeddigs_of_word.append(word_to_vec_model[token])
            else:
                out_of_vocab_embedding = self.get_out_of_vocab_embedding(token=token, word_to_vec_model_id=i)
                embeddigs_of_word.append(out_of_vocab_embedding)

        return embeddigs_of_word

    def get_part_of_speech_embedding(self, pos_tag):
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

    def merge_feature_embeddings(self):
        pass

    def compute_sequence_embedding(self, txt, offset_tuple,
                                   is_entity, nlp_span):
        """

        :param txt: Text from which substrings are extracted.
        :param offset_tuple: Tuple containing offset information about the substring to process.
        :param is_entity: Flag indicating whether relevant substring represents an entity.
        :param nlp_span: Container containing all NLP features for relevant substring.
        :rtype: list
        [  [ [token-1_model-1 feature embedding],...,[token-1_model-n feature embedding] ], ... ,
        [ [token-k_model-1 feature embedding],...,[token-k_model-n feature embedding] ]  ]
        """

        use_part_of_speech_embedding = False

        if PART_OF_SPEECH_VEC_DIM in self.features_spec_dict:
            use_part_of_speech_embedding = True

        if is_entity:
            start = offset_tuple[0]
            end = offset_tuple[1]
            entity = txt[start:end]
            seq_embedding = self.get_sequence_embedding_for_entity(entity=entity, nlp_span=nlp_span,
                                                                   use_part_of_speech_embedding=
                                                                   use_part_of_speech_embedding)
        else:
            tokens = [token.text for token in nlp_span]
            token_embeddings = [self.get_embeddings_for_token(token) for token in tokens]
            seq_embedding = token_embeddings

            if use_part_of_speech_embedding:
                for token in tokens:
                    part_of_speech_embedding = self.get_part_of_speech_embedding(pos_tag=token)
                    # Copy POS-tag embedding if several word2Vec models should be used
                    n_times_part_of_speech_embedding = np.repeat(a=[part_of_speech_embedding],
                                                                 repeats=len(self.word_to_vec_models), axis=0).tolist()
                    # TODO: Merge features

        return seq_embedding

    def add_padding_to_embedding(self, seq_embedding):
        """
        Adds zero padding vectors to the left and right of an embedded sequence if number of embedding tokens
        is smaller than self.max_num_utter_tokens.
        :param seq_embedding: Complete embedded sequence of a text
        :rtype: list
        """
        num_tokens = len(seq_embedding)
        left_padding_size = (self.max_num_utter_tokens - num_tokens) // 2
        right_padding_size = self.max_num_utter_tokens - num_tokens - left_padding_size

        shape_for_left_padding = (
            left_padding_size, len(self.word_to_vec_models), self.features_spec_dict[WORD_VEC_DIM])
        shape_for_right_padding = (
            right_padding_size, len(self.word_to_vec_models), self.features_spec_dict[WORD_VEC_DIM])

        left_padding = np.zeros(shape=shape_for_left_padding).tolist()
        right_padding = np.zeros(shape=shape_for_right_padding).tolist()

        seq_padded = left_padding + seq_embedding + right_padding

        return seq_padded

    def create_tensor(self, embedded_sequence):
        """
        Creates a tensor representation of the embedded sequence with the expected shape.
        Each row represents represents an embedded word or entity. The number of columns indicate the embedding
        dimension. If more than one word2Vec model is used, an additional layer will be added representing the channel
        dimension (like in a picture). Then in each channel one representation of a word/entity is saved.
        :param embedded_sequence: List containing the embeddings of the sequence. Order matters.
        :rtype: numpy.array() with shape= (#-tokens, word vector dimension, #-word2Vec models)
        """

        num_rows = self.max_num_utter_tokens
        num_columns = self.features_spec_dict[WORD_VEC_DIM]
        num_word_to_vec_models = len(self.word_to_vec_models)

        if num_word_to_vec_models > 1:
            num_channels = num_word_to_vec_models
            tensor_shape = (num_rows, num_columns, num_word_to_vec_models)
            # TODO: Replace implementation in next version
            tensor_embedding = [np.stack(embeddings_of_tokens, axis=-1) for embeddings_of_tokens in embedded_sequence]
            tensor_embedding = np.array(tensor_embedding)
        else:
            tensor_shape = (num_rows, num_columns, num_word_to_vec_models)
            embedded_sequence = np.array(embedded_sequence, dtype=float)
            tensor_embedding = np.reshape(embedded_sequence, newshape=tensor_shape)

        return tensor_embedding

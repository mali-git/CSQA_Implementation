import logging

import bisect
import numpy as np
from gensim.models import KeyedVectors

from utilities.corpus_preprocessing.load_dialogues import load_data_from_json_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from utilities.constants import WORD_VEC_DIM, CSQA_UTTERANCE, CSQA_ENTITIES_IN_UTTERANCE


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
        if WORD_VEC_DIM in features_spec_dict:
            assert (word_to_vec_dict is not None)

        self.word_to_vec_models = self.load_word_2_vec_models(word_to_vec_dict=word_to_vec_dict)
        self.entity_id_to_label_dict = self.load_entity_to_label_mapping(path_to_entity_id_to_label_mapping)
        self.predicate_id_to_label_dict = self.load_entity_to_predicate_mapping(path_to_predicate_id_to_label_mapping)
        self.kg_embeddings_dict = self.load_kg_embeddings(path_to_kb_embeddings=path_to_kb_embeddings)
        self.max_num_utter_tokens = max_num_utter_tokens
        self.features_spec_dict = features_spec_dict

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
            question_offsets_info_dict = self.get_offsets_of_parts_in_utterance(question)
            answer_offsets_info_dict = self.get_offsets_of_parts_in_utterance(answer)

            # Step 2: Compute tensor embedding for utterance
            embedded_question = self.compute_tensor_embedding(utterance_dict=question)
            embedded_answer = self.compute_tensor_embedding(utterance_dict=answer)

            # Step 3: Create training instance
            training_instance_dict = self.create_instance_dict(is_training_instance=True)
            training_instance_dicts.append(training_instance_dict)

        return training_instance_dict

    def get_offsets_of_parts_in_utterance(self, utterance_dict):
        """
        Returns sorted (increasing) offsets of utterance-parts based on mentioned entities
        :param utterance_dict: Dictionary containing all information about passed utterance
        :rtype: dict
        """
        utterance = utterance_dict[CSQA_UTTERANCE]
        entities_in_utterance = utterance_dict[CSQA_ENTITIES_IN_UTTERANCE]
        start_offsets = []
        end_offsets = []

        for entity_in_utterance in entities_in_utterance:
            # Get offsets of entity
            start = utterance.find(entity_in_utterance)

            if start == -1:
                # Entity not found
                log.info("Entity %s not found in utterance %s" % (entity_in_utterance, utterance))
                continue

            end = start + len(entity_in_utterance)

            # Get updated list of ofsets. Function handles overlaps.
            updated_start_offsets, updated_end_offsets = self.save_insertion_of_offsets(start_offsets, end_offsets,
                                                                                        start, end)


    def compute_tensor_embedding(self, utterance_dict, utterance_offsets_info_dict):
        pass

    def create_instances_for_prediction(self):
        pass

    def create_instance_dict(self, is_training_instance):
        pass

    def save_insertion_of_offsets(self, start_offsets, end_offsets, start, end):
        # Get the smallest value where start can be inserted in the sorted list
        index_of_smallest = bisect.bisect(start_offsets, start)
        indices_of_overlapped_offsets = []
        save_start_offsets = []
        save_end_offsets = []

        if index_of_smallest == 0:
            log.info(
                "start is not in start offsets --> All existing end offsets smaller than current end are overlapped")
            # start is not in start offsets --> All existing end offsets smaller than current end are overlapped
            temp_np_array = np.array(end_offsets, dtype=int)
            current_indices_of_none_overlapped_offsets = ((temp_np_array > end) * 1).nonzero()
            save_start_offsets = start_offsets[current_indices_of_none_overlapped_offsets]
            save_end_offsets = end_offsets[current_indices_of_none_overlapped_offsets]


        elif index_of_smallest == len(start_offsets):
            # start is greater then all current values --> there are no overlapping offsets
            save_start_offsets = start_offsets
            save_end_offsets = end_offsets

        else:
            # start lies between smallest and biggest current start

            # Step 1: Get indices of end positions which are bigger or equal to current end
            temp_np_array = np.array(end_offsets, dtype=int)
            current_end_offsets_indices_bigger_or_equal = ((temp_np_array >= end) * 1).nonzero()
            non_over_lapping_offsets = np.ones(shape=len(start_offsets)).tolist()
            insertion_allowed = False

            # Step 2: Check relevant start indices
            # relevant_start_indices = start_offsets[current_end_offsets_indices_bigger_or_equal]
            for relevant_index in current_end_offsets_indices_bigger_or_equal:
                current_start = start_offsets[relevant_index]
                if current_start <= start:
                    # Case: ['Chancellor of Germany'] Insert: 'Germany' -> Don't insert
                    continue
                else:
                    # Case: ['Germany'] Insert: 'Chancellor of Germany' -> Insert
                    non_over_lapping_offsets[relevant_index] = 0
                    insertion_allowed = True

            # Step 3: Determine safe start offsets
            save_start_offsets = np.array(start_offsets[non_over_lapping_offsets], dtype=int).nonzero().tolist()
            # Insert new start if possible
            if insertion_allowed:
                bisect.insort_left(save_start_offsets, start)
                bisect.insort_left(save_end_offsets, end)

        return save_start_offsets, save_end_offsets


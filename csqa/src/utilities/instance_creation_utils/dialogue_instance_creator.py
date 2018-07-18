import logging
import pickle
from collections import OrderedDict

import numpy as np
import spacy
from gensim.models import KeyedVectors

from utilities.constants import CSQA_UTTERANCE, CSQA_ENTITIES_IN_UTTERANCE, UNKNOWN_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    KG_WORD, PADDING_TOKEN, TOKEN_IDS, INSTANCE_ID, CSQA_QUES_TYPE_ID
from utilities.corpus_preprocessing_utils.load_dialogues import load_data_from_json_file
from utilities.corpus_preprocessing_utils.text_manipulation_utils import save_insertion_of_offsets, mark_parts_in_text, \
    compute_nlp_features

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DialogueInstanceCreator(object):

    def __init__(self, max_num_utter_tokens, max_dialogue_context_length,
                 path_to_entity_id_to_label_mapping, path_to_property_id_to_label_mapping, ctx_vocab_freq_dict,
                 response_vocab_freq_dict, word_to_vec_dict, path_to_kb_entities_embeddings,
                 path_to_wikidata_triples, seed, min_count_n_gram_matching=1000):

        # Set seed to allow reproducibility
        self.seed = seed
        np.random.seed(self.seed)

        self.word_to_vec_model = self._load_word_to_vec_model(word_to_vec_dict=word_to_vec_dict)
        self.kg_entities_embeddings_dict = self._load_kg_embeddings(
            path_to_kb_embeddings=path_to_kb_entities_embeddings)

        self.wiki_data_triples = np.loadtxt(fname=path_to_wikidata_triples, dtype=str,
                                            comments='@Comment@ Subject Predicate Object')

        self.entity_kg_id_to_label_dict = load_data_from_json_file(path_to_entity_id_to_label_mapping)
        self.entity_label_to_kg_id_dict = {v: k for k, v in self.entity_kg_id_to_label_dict.items()}
        self.predicate_kg_id_to_label_dict = load_data_from_json_file(path_to_property_id_to_label_mapping)
        self.predicate_label_to_kg_id_dict = {v: k for k, v in self.predicate_kg_id_to_label_dict.items()}

        self.modified_items, self.modified_items_id = self._build_modified_wikidata_items_dict()

        self.min_count_n_gram_matching = min_count_n_gram_matching
        self.ctx_vocab_freq_dict = ctx_vocab_freq_dict
        self.max_num_utter_tokens = max_num_utter_tokens
        # Context of length is defined by a user utterance followed by a system utterance
        self.max_dialogue_context_length = max_dialogue_context_length

        # Note: Flag is important
        self.ctx_token_to_embeddings, self.ctx_word_to_id = self._initialize_token_mappings(
            vocab_freqs=ctx_vocab_freq_dict, is_ctx_vocab=True)

        # Note: Flag is important
        self.response_token_to_embeddings, self.response_word_to_id = self._initialize_token_mappings(
            vocab_freqs=response_vocab_freq_dict, is_ctx_vocab=False)

        self.ctx_num_trainable_toks = 4
        self.response_num_trainable_toks = 5

        self.responses_vocab = OrderedDict()
        self.nlp_parser = spacy.load('en')
        self.current_file_in_progress = None

        self._initialize_token_mappings(vocab_freqs=response_vocab_freq_dict, is_ctx_vocab=False)
        self.entity_to_id, self.rel_to_id = self._initilaize_kg_item_mappings(triples=self.wiki_data_triples)

    def _load_kg_embeddings(self, path_to_kb_embeddings):
        """
        Load dict containing the KG item id's as keys and the embeddings as values
        :param path_to_kb_embeddings: Path to the serialized dict
        :rtype: dict
        """
        with open(path_to_kb_embeddings, 'rb') as f:
            kg_items_to_embeddings_dict = pickle.load(f)
            return kg_items_to_embeddings_dict

    def _add_kg_embeddings_to_vocab(self, token_to_embeddings):
        """
        Add the entity id's and their embeddings to the passed dict
        :param token_to_embeddings:
        :param token_to_id_dict:
        :rtype: dict, dict
        """
        # Add KG embeddings
        # no assignment nedded for 'token_to_embeddings' since update() returns None
        token_to_embeddings.update(self.kg_entities_embeddings_dict)

        return token_to_embeddings

    def _initialize_token_mappings(self, vocab_freqs, is_ctx_vocab):
        """
        Extract tokens from vocab_freqs, and the embeddings from the word2Vec model, and create corresponding mappings
        :param vocab_freqs: Dict containing tokens as keys and their frequencies in the corpus as values
        :param is_ctx_vocab: Flag indication whether the vocabulary is extracted from context utterances
        :rtype: dict, dict
        """
        token_to_embeddings = OrderedDict()
        vec_dim = self.word_to_vec_model.vector_size
        token_to_embeddings[SOS_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[EOS_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[UNKNOWN_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[PADDING_TOKEN] = np.zeros(shape=(vec_dim,))

        if is_ctx_vocab == False:
            token_to_embeddings[KG_WORD] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))

        # TODO: Improove
        for word, _ in vocab_freqs.items():
            if word not in token_to_embeddings:
                if word in self.word_to_vec_model:
                    token_to_embeddings[word] = self.word_to_vec_model[word]

        if is_ctx_vocab:
            # KG entities are added only in context vocabulary
            # In response vocabulary the special token KG_WORD is used for entities
            token_to_embeddings = self._add_kg_embeddings_to_vocab(
                token_to_embeddings=token_to_embeddings)

        token_to_id_dict = {token: id for id, token in enumerate(token_to_embeddings.keys())}

        return token_to_embeddings, token_to_id_dict

    def _load_entity_to_label_mapping(self, path_to_entity_id_to_label_mapping):
        """
        Load dict containing entity id's as keys and the corresponding labels as values
        :param path_to_entity_id_to_label_mapping:
        :rtype: dict
        """

        return load_data_from_json_file(path_to_entity_id_to_label_mapping)

    def _load_word_to_vec_model(self, word_to_vec_dict):
        """
        Load pretrained word2Vec model
        :param word_to_vec_dict:
        :return:
        """
        path = word_to_vec_dict['Path']
        is_c_format = word_to_vec_dict['is_c_format']
        is_binary = word_to_vec_dict['is_binary']

        if is_c_format:
            word_to_vec_model = KeyedVectors.load_word2vec_format(path, binary=is_binary)
        else:
            word_to_vec_model = KeyedVectors.load(fname_or_handle=path)

        return word_to_vec_model

    def _get_offsets_of_relevant_entities_in_utterance(self, utterance_dict):
        """
        Returns sorted (increasing) offsets of utterance-parts based on mentioned entities
        :param utterance_dict: Dictionary containing all information about passed utterance
        :rtype: list, list
        """
        start_offsets, end_offsets = [], []

        # No entities in utterance
        if CSQA_ENTITIES_IN_UTTERANCE not in utterance_dict:
            utterance_dict[CSQA_ENTITIES_IN_UTTERANCE] = []
            return start_offsets, end_offsets

        utterance = utterance_dict[CSQA_UTTERANCE]

        ids_of_entities_in_utterance = utterance_dict[CSQA_ENTITIES_IN_UTTERANCE]
        entities_in_utterance = [self.entity_kg_id_to_label_dict[entity_id] for entity_id in
                                 ids_of_entities_in_utterance]

        for entity_in_utterance in entities_in_utterance:
            # Get offsets of entity
            start = utterance.find(entity_in_utterance)

            if start == -1:
                # Entity not found
                log.info("Entity '%s' not found in utterance '%s' contained in file '%s'" % (
                    entity_in_utterance, utterance, self.current_file_in_progress))
                continue

            end = start + len(entity_in_utterance)

            # Get updated list of ofsets. Function handles overlaps.
            start_offsets, end_offsets = save_insertion_of_offsets(start_offsets, end_offsets,
                                                                   start, end)

        return start_offsets, end_offsets

    def _map_utter_toks_to_ids(self, utterance_dict, utterance_offsets_info_dict, nlp_spans, is_reponse_utter):
        """
        Return for tokens and KG entities of an utterance their corresponding id's
        :param utterance_dict: Dictionary containing all information about passed utterance
        :param utterance_offsets_info_dict: Dict: Keys are tuples (start,end) and values are boolean flags (is_entity)
        :param nlp_spans: Contains each token NLP features
        :param is_reponse_utter: Flag
        :rtype: list
        """

        utterance = utterance_dict[CSQA_UTTERANCE]
        counter = 0

        token_ids = []
        for offset_tuple, is_entity in utterance_offsets_info_dict.items():
            nlp_span = nlp_spans[counter]
            token_ids += self._determine_token_ids(txt=utterance, offset_tuple=offset_tuple, is_entity=is_entity,
                                                   nlp_span=nlp_span, is_reponse_utter=is_reponse_utter)

            counter += 1

        return token_ids

    def _apply_instance_creation_steps(self, utterance_dict, is_reponse_utter):
        utterance_txt = utterance_dict[CSQA_UTTERANCE]

        # Step 1: Get offsets of utterance-parts based on mentioned entites
        relevant_entity_start_offsets_in_utterance, relevant_entity_end_offsets_in_utterance = \
            self._get_offsets_of_relevant_entities_in_utterance(
                utterance_dict)

        utterance_offset_info_dict = mark_parts_in_text(
            start_offsets_entities=relevant_entity_start_offsets_in_utterance,
            end_offsets_entities=relevant_entity_end_offsets_in_utterance,
            text=utterance_txt)

        # Step 2: Compute NLP features
        utterance_nlp_spans = compute_nlp_features(txt=utterance_txt,
                                                   offsets_info_dict=utterance_offset_info_dict)

        utterance_tok_ids = self._map_utter_toks_to_ids(utterance_dict=utterance_dict,
                                                        utterance_offsets_info_dict=utterance_offset_info_dict,
                                                        nlp_spans=utterance_nlp_spans,
                                                        is_reponse_utter=is_reponse_utter)

        padded_utter_tok_id = self.add_utter_padding(utter_tok_ids=utterance_tok_ids, is_reponse_utter=is_reponse_utter)

        return padded_utter_tok_id

    def _determine_token_ids(self, txt, offset_tuple,
                             is_entity, nlp_span, is_reponse_utter):
        """

        :param txt: Utterance
        :param offset_tuple: Tuple indicating start and end of part
        :param is_entity: Flag indicating whether part is corresponding to an entity
        :param nlp_span: NLP features of part
        :param is_reponse_utter: Flag indicating wheter utterance is a response
        :rtype: list
        """

        if is_reponse_utter:
            word_to_id = self.response_word_to_id
        else:
            word_to_id = self.ctx_word_to_id

        if is_entity:
            start = offset_tuple[0]
            end = offset_tuple[1]
            # Entity don't transform to lowercase
            entity = txt[start:end]
            entity_id = self.entity_label_to_kg_id_dict[entity]

            return [self._get_token_id_for_entity(entity_id, is_reponse_utter)]

        else:
            return [word_to_id[token.lower_] if token.lower_ in word_to_id else word_to_id[UNKNOWN_TOKEN] for token in
                    nlp_span]

    def _get_token_id_for_entity(self, entity_id, is_reponse_utter):
        """
        Retruns token id (not KG id) for entity
        :param entity_id: KG id of entity
        :param is_reponse_utter: Flag indicating wheter utterance is a response
        :rtype: int
        """

        if is_reponse_utter:
            id = self.response_word_to_id[KG_WORD]
        else:

            if entity_id in self.ctx_word_to_id:
                id = self.ctx_word_to_id[entity_id]
            else:
                id = self.ctx_word_to_id[UNKNOWN_TOKEN]

        return id

    def create_training_instances(self, dialogue, file_id):
        """

        :param dialogue:
        :param file_id:
        :rtype: list, dict, np.array
        """
        instances_of_dialogue, target_inst, relevant_kg_triples = self._create_instances(dialogue, file_id,
                                                                                         is_training_mode=True)
        return instances_of_dialogue, target_inst, relevant_kg_triples

    def create_inference_instances(self, dialogue, file_id):
        """

        :param dialogue:
        :param file_id:
        :rtype: list, np.array
        """
        instances_of_dialogue, _, relevant_kg_triples = self._create_instances(dialogue, file_id,
                                                                               is_training_mode=False)
        return instances_of_dialogue, relevant_kg_triples

    def _create_instances(self, dialogue, file_id, is_training_mode):
        """

        :param dialogue:
        :param file_id:
        :param is_training_mode:
        :rtype: list, dict, np.array
        """
        dialogue_instance = []

        # Set state
        self.current_file_in_progress = file_id

        if len(dialogue) > self.max_dialogue_context_length * 2:
            dialogue = dialogue[-(self.max_dialogue_context_length * 2):]

        context = dialogue[:-1]
        response = dialogue[-1]

        counter = 0

        for utter_dict in context:
            utter_token_ids = self._apply_instance_creation_steps(utterance_dict=utter_dict, is_reponse_utter=False)
            instance_id = file_id + '_' + str(counter)
            new_inst = self._create_single_instance(utter_dict=utter_dict, utter_token_ids=utter_token_ids,
                                                    instance_id=instance_id)
            dialogue_instance.append(new_inst)
            counter += 1

        utter_token_ids = self._apply_instance_creation_steps(utterance_dict=response, is_reponse_utter=True)
        instance_id = file_id + '_' + str(counter)

        if is_training_mode:
            target_input, target_output = self._create_target_instance(utter_dict=response,
                                                                       utter_token_ids=utter_token_ids,
                                                                       instance_id=instance_id)
            dialogue_instance.append(target_input)
            target = target_output
        else:
            new_inst = self._create_single_instance(utter_dict=response, utter_token_ids=utter_token_ids,
                                                    instance_id=instance_id)
            dialogue_instance.append(new_inst)
            target = None

        # Extract relevant KG triples
        relevant_kg_triples = self._extract_relevant_kg_triples(utter_dict=context[-1])

        return dialogue_instance, target, relevant_kg_triples

    def _create_single_instance(self, utter_dict, utter_token_ids, instance_id):
        new_inst = OrderedDict()
        new_inst[INSTANCE_ID] = instance_id
        new_inst[CSQA_UTTERANCE] = utter_dict[CSQA_UTTERANCE]
        new_inst[TOKEN_IDS] = utter_token_ids
        new_inst[CSQA_ENTITIES_IN_UTTERANCE] = utter_dict[CSQA_ENTITIES_IN_UTTERANCE]

        if CSQA_QUES_TYPE_ID in utter_dict:
            # Utternace is a question
            new_inst[CSQA_QUES_TYPE_ID] = int(utter_dict[CSQA_QUES_TYPE_ID])

        return new_inst

    def _create_target_instance(self, utter_dict, utter_token_ids, instance_id):
        # Remove last token: Make sure that it is a <padding> token
        utter_token_ids = utter_token_ids[:-1]
        # Add <sos> token at the beginning
        target_in_tok_ids = [self.response_word_to_id[SOS_TOKEN]] + utter_token_ids
        target_out_tok_ids = utter_token_ids + [self.response_word_to_id[EOS_TOKEN]]

        target_input_inst = OrderedDict()
        target_input_inst[INSTANCE_ID] = instance_id
        target_input_inst[CSQA_UTTERANCE] = utter_dict[CSQA_UTTERANCE]
        target_input_inst[TOKEN_IDS] = target_in_tok_ids
        target_input_inst[CSQA_ENTITIES_IN_UTTERANCE] = utter_dict[CSQA_ENTITIES_IN_UTTERANCE]

        target_output_inst = OrderedDict()
        target_output_inst[INSTANCE_ID] = instance_id
        target_output_inst[CSQA_UTTERANCE] = utter_dict[CSQA_UTTERANCE]
        target_output_inst[TOKEN_IDS] = target_out_tok_ids
        target_output_inst[CSQA_ENTITIES_IN_UTTERANCE] = utter_dict[CSQA_ENTITIES_IN_UTTERANCE]

        return target_input_inst, target_output_inst

    def add_utter_padding(self, utter_tok_ids, is_reponse_utter):
        """
        Add padding to the right side to id list
        :param utter_tok_ids: List of token ids
        :param is_reponse_utter: Flag indicating whether part is corresponding to an entity
        :rtype: list
        """
        num_tokens = len(utter_tok_ids)
        right_padding_size = (self.max_num_utter_tokens - num_tokens)
        if is_reponse_utter:
            padding = [self.response_word_to_id[PADDING_TOKEN] for i in range(right_padding_size)]
        else:
            padding = [self.ctx_word_to_id[PADDING_TOKEN] for i in range(right_padding_size)]

        return utter_tok_ids + padding

    def _build_modified_wikidata_items_dict(self):
        kg_items = list(self.entity_kg_id_to_label_dict.values()) + list(self.predicate_kg_id_to_label_dict.values())
        kg_ids = list(self.entity_kg_id_to_label_dict.keys()) + list(self.predicate_kg_id_to_label_dict.keys())

        items = []
        item_ids = []

        for i, value in enumerate(kg_items):
            id = kg_ids[i]
            parts = value.split()
            items += parts
            item_ids += [id for i in range(len(parts))]

        items = np.array(items, dtype=np.str)
        item_ids = np.array(item_ids, dtype=np.str)

        return items, item_ids

    def _extract_relevant_kg_triples(self, utter_dict):
        """
        Extract for utter relevant triples from KG based on n-gram matching
        :param utter_dict: Dictionary containing all information about passed utterance
        :rtype: np.array
        """
        data = self.wiki_data_triples
        utter_txt = utter_dict[CSQA_UTTERANCE]
        doc = self.nlp_parser(u'%s' % (utter_txt))
        tokens = [tok.lower_ for tok in doc]
        relevant_toks = [tok for tok in tokens if tok in self.ctx_vocab_freq_dict and self.ctx_vocab_freq_dict[
            tok] <= self.min_count_n_gram_matching]
        relevant_toks = np.array(relevant_toks, dtype=np.str)
        relevant_items_indices = np.nonzero((np.isin(self.modified_items, relevant_toks) * 1.0))

        relevant_ids = np.unique(self.modified_items_id[relevant_items_indices])

        subject_ids = self.wiki_data_triples[:, 0:1]
        predicate_ids = self.wiki_data_triples[:, 1:2]
        object_ids = self.wiki_data_triples[:, 2:3]

        relevant_subj_indices = np.nonzero((np.isin(subject_ids, relevant_ids) * 1.0))
        relevant_pred_indices = np.nonzero((np.isin(predicate_ids, relevant_ids) * 1.0))
        relevant_obj_indices = np.nonzero((np.isin(object_ids, relevant_ids) * 1.0))

        relevant_triples_indices = np.unique(
            np.concatenate([relevant_subj_indices, relevant_pred_indices, relevant_obj_indices], axis=-1))

        relevant_triples = data[relevant_triples_indices]
        relevant_subj_ids = np.vectorize(self.entity_to_id.get)(relevant_triples[:, 0:1])
        relevant_predicate_ids = np.vectorize(self.rel_to_id.get)(relevant_triples[:, 1:2])
        relevant_obj_ids = np.vectorize(self.entity_to_id.get)(relevant_triples[:, 2:3])

        relevant_triples = np.concatenate([relevant_subj_ids, relevant_predicate_ids, relevant_obj_ids], axis=-1)

        return relevant_triples

    def _initilaize_kg_item_mappings(self, triples):
        subjects = triples[:, 0:1]
        predicates = triples[:, 1:2]
        objects = triples[:, 2:3]

        entities = list(set(np.ndarray.flatten(np.concatenate([subjects, objects])).tolist()))
        relations = list(set(np.ndarray.flatten(predicates).tolist()))
        entity_to_id = {value: key for key, value in enumerate(entities)}
        rel_to_id = {value: key for key, value in enumerate(relations)}

        return entity_to_id, rel_to_id

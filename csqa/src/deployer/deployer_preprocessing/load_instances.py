import logging
from collections import OrderedDict

import click

from utilities.constants import PATH_TO_INPUT, MAX_NUM_UTTER_TOKENS, WORD_VEC_DIM, PATH_TO_ENTITY_MAPPING_FILE, \
    PATH_TO_WORD_TO_WEC_MODELS, WORD_TO_VEC_FLAGS_FOR_BINARY_FORMAT, WORD_TO_VEC_FLAGS_FOR_C_FORMAT
from utilities.corpus_preprocessing_utils.load_dialogues import retrieve_dialogues
from utilities.deployer_utils import get_parameters_from_configuration_file
from utilities.instance_creation_utils.feature_utils import get_feature_specification_dict
from utilities.instance_creation_utils.utterance_to_tensor_embeddings_creator import Utterance2TensorCreator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@click.command()
@click.option('-conf', help='path to configuration file', required=True)
def main(conf):
    path_to_config = conf
    # Get configuration parameters
    config_param_dict = get_parameters_from_configuration_file(path_to_config=path_to_config)

    # ------------------Assign parameters------------------
    input_direc = config_param_dict[PATH_TO_INPUT]
    max_num_utter_tokens = int(config_param_dict[MAX_NUM_UTTER_TOKENS])
    word_vec_dim = int(config_param_dict[WORD_VEC_DIM])
    path_to_entity_mapping_file = config_param_dict[PATH_TO_ENTITY_MAPPING_FILE]
    word_to_vec_model_paths = [path for path in config_param_dict[PATH_TO_WORD_TO_WEC_MODELS].split('@@sep@@')]
    word_to_vec_file_flags_for_binary_format = [int(file_format) for file_format in
                                                config_param_dict[WORD_TO_VEC_FLAGS_FOR_BINARY_FORMAT].split(',')]
    word_to_vec_file_flags_for_c_format = [int(file_format) for file_format in
                                           config_param_dict[WORD_TO_VEC_FLAGS_FOR_C_FORMAT].split(',')]
    word_to_vec_model_dict = OrderedDict()

    for i, path in enumerate(word_to_vec_model_paths):
        is_c_format = word_to_vec_file_flags_for_c_format[i]
        is_binary_format = word_to_vec_file_flags_for_binary_format[i]
        word_to_vec_model_dict[path] = [is_c_format, is_binary_format]

    feature_specification_dict = get_feature_specification_dict(word_vec_dim=word_vec_dim,
                                                                position_vec_dim=None,
                                                                part_of_speech_vec_dim=None)

    # ------------------Load dialogues------------------
    dialogues_data_dict = retrieve_dialogues(input_directory=input_direc)

    # Create training instances
    instance_creator = Utterance2TensorCreator(max_num_utter_tokens=max_num_utter_tokens,
                                               features_spec_dict=feature_specification_dict,
                                               path_to_entity_id_to_label_mapping=path_to_entity_mapping_file,
                                               word_to_vec_dict=word_to_vec_model_dict,
                                               path_to_kb_embeddings=None)

    training_instance_dicts = [instance_creator.create_training_instances(dialogue=dialogue, file_id=key) for
                               key, dialogue in dialogues_data_dict.items()]


if __name__ == '__main__':
    main()

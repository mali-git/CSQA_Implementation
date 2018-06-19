
# Instance related
INSTANCE_ID = 'instance_id'
ENTITY_ID = 'entity_id'
PREDICATE_IDS_QUESTION = 'predicate_ids_question'
PREDICATE_IDS_ANSWER = 'predicate_ids_answer'
QUESTION_ENTITIES = 'entities_in_question'
QUESTION_UTTERANCE = 'question_utterance'
QUESTION_UTTERANCE_EMBEDDED = 'question_utterance_embedded'
ANSWER_ENTITIES = 'entities_in_answer'
ANSWER_UTTERANCE = 'answer_utterance'
ANSWER_UTTERANCE_EMBEDDED = 'answer_utterance_embedded'

# Given keys in the training corpus
CSQA_ALL_ENTITIES = 'all_entities'
CSQA_SPEAKER = 'speaker'
CSQA_ENTITIES_IN_UTTERANCE = 'entities_in_utterance'
CSQA_UTTERANCE = 'utterance'
CSQA_ACTIVE_SET = 'active_set'
CSQA_DESCRIPTION = 'description'
CSQA_QUES_TYPE_ID = 'ques_type_id'
CSQA_QUESTION_TYPE = 'question-type'
CSQA_RELATIONS = 'relations'
CSQA_TYPE_LIST = 'type_list'

# Features (position features, part-of-speech features etc.) related
WORD_VEC_DIM = 'word_vec_dim'
POSITION_VEC_DIM = 'position_vec_dim'
PART_OF_SPEECH_VEC_DIM = 'part_of_speech_vec_dim'

# Needed for configuration
PATH_TO_INPUT = 'path_to_input'
PATH_TO_MODEL_DIREC = 'path_to_model_directory'
PATH_TO_WORD_TO_WEC_MODELS = 'path_to_word_vec_models'
PATH_TO_ENTITY_MAPPING_FILE = 'path_to_entity_mapping_file'
MAX_NUM_UTTER_TOKENS = 'max_num_utter_tokens'
WORD_TO_VEC_FLAGS_FOR_BINARY_FORMAT = 'word_to_vec_flags_for_binary_format'
WORD_TO_VEC_FLAGS_FOR_C_FORMAT = 'word_to_vec_flags_for_c_format'
BATCH_SIZE = 'batch_size'

# Additional constants
EMBEDDED_SEQUENCES = 'embedded_sequences'
EMBEDDED_RESPONSES = 'embedded_responses'
DIALOGUES = 'dialogues'
RESPONSES = 'responses'
ENCODER_VOCABUALRY_SIZE = 'encoder_vocabulary_size'
DECODER_VOCABUALRY_SIZE = 'decoder_vocabulary_size'
LOGITS = 'logits'
WORD_IDS = 'word_ids'
WORD_PROBABILITIES = 'word_probabilities'

# Hierarchical encoder related constants
NUM_UNITS_HRE_UTTERANCE_CELL = 'num_units_hre_utterance_cell'
NUM_UNITS_HRE_CONTEXT_CELL = 'num_units_hre_context_cell'

# Key-Value Memory Network related constants
NUM_HOPS = 'num_hops'
KEY_CELLS = 'key_cells'
VALUE_CELLS = 'value_cells'

# Optimizers
OPTIMIZER = 'optimizer'
ADADELTA = 'Adadelta'
ADAGRAD ='Adagrad'
ADAGRAD_DA = 'AdagradDA'
ADAM = 'Adam'
RMS_PROP = 'RMSProp'

# Constants refering to hyperparamters
LEARNING_RATE = 'learning_rate'

# Serving keys needed by estimator
DEFAULT_SERVING_KEY = "serving_default"
CLASSIFY_SERVING_KEY = 'classification'
PREDICT_SERVING_KEY = 'predict'

# Vocabulary related keys
TARGET_SOS_ID = 'target_sos_id'
TARGET_EOS_ID = 'target_eos_id'
TARGET_VOCAB_SIZE = 'target_vocab_size'
ENCODER_NUM_TRAINABLE_TOKENS = 'encoder_number_of_trainable_tokens'
DECODER_NUM_TRAINABLE_TOKENS = 'decoder_number_of_trainable_tokens'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNKNOWN_TOKEN = '<unk>'
KG_WORD = '<kg_word>'
PADDING_TOKEN = '<padding>'

# Configuration related
UTILITIES_PATH = 'utilities'
READER = 'reader'
WROC_READER = 'WROCReader'
TRANS_E = 'TransE'
TRANS_H = 'TransH'
CLASS_NAME = 'class_name'


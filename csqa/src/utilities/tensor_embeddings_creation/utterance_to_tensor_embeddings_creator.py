
import gensim

class Utterance2TensorCreator(object):

    def __init__(self, word_to_vec_dict, max_num_utter_tokens, path_to_kb_embeddings):
        """
        :param word_to_vec_dict: Dictionary containing as keys the paths to the word2Vec models. Values indicate
        wheter a model is in binary format or not.
        :param max_num_utter_tokens: Maximum length (in tokens) of an utterance
        :param path_to_kb_embeddings: Path to KB embeddings
        """
        self.word_to_vec_models = self.load_word_to_vec_models()
        self.kg_embeddings_dict = dict()
        self.max_num_utter_tokens = max_num_utter_tokens


    def load_word_to_vec_models(self, word_to_vec_dict):
        pass

    def load_kg_embeddings(self, path_to_kb_embeddings):
        pass

    def compute_instances(self, compute_training_instances):
        pass

    def compute_training_instances(self):
        return self.compute_instances(compute_training_instances=True)

    def compute_instances_for_prediction(self):
        return self.compute_instances(compute_training_instances=False)
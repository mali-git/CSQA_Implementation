
class KGEmbeddingCreatorContainer(object):

    def __init__(self, corpus_reader, training_instance_creator, embedding_model):
        """

        :param corpus_reader:
        :param training_instance_creator:
        :param embedding_model:
        """
        self.corpus_reader = corpus_reader
        self.training_instance_creator = training_instance_creator
        self.embedding_model = embedding_model
        self.entity_to_id = None
        self.relation_to_id = None

    def train_kg_embeddings(self):
        self.entity_to_id, self.relation_to_id = None
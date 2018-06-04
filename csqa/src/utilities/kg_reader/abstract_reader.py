
import abc

class KGReader(object):

    def __init__(self, entities_min_count=None, relation_min_count=None):
        self.entities_min_count = entities_min_count
        self.relation_min_count = relation_min_count
        self.entities_to_freq = None
        self.relations_to_freq = None

    @abc.abstractmethod
    def read_kg(self, path_to_kg_file):
        return
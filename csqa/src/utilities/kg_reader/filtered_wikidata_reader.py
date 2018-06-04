import json
import multiprocessing
from multiprocessing import Pool

from utilities.general_utils import split_list_in_chunks
from utilities.kg_reader.abstract_reader import KGReader


class FilteredWikiDataReader(KGReader):

    def __init__(self, entities_min_count=None, relation_min_count=None):
        super(FilteredWikiDataReader, self).__init__(entities_min_count, relation_min_count)

    def read_kg(self, path_to_kg_file):
        with open(path_to_kg_file, 'r', encoding='utf-8') as fp:
            dict = json.load(fp)
        return dict

    def _extract(self, argument_tuple):
        """

        :param argument_tuple:
        :return:
        """
        triples = []
        keys_of_kg_dict, kg = argument_tuple
        for subject in keys_of_kg_dict:
            value_dict = kg[subject]
            for predicate, object_list in value_dict.items():
                if not object_list:
                    continue
                for object in object_list:
                    triples.append((subject, predicate, object))

        return triples

    def extract_triples(self, kg, is_mulit_processing_mode=True, num_processes=None):
        """

        :param kg:
        :param is_mulit_processing_mode:
        :param num_processes:
        :return:
        """

        if is_mulit_processing_mode == False and num_processes != None:
            raise Exception("When defining 'num_processes' then 'is_mulit_processing_mode' must be set to 'True'")

        keys = list(kg.keys())

        if is_mulit_processing_mode == False:
            return self._extract(argument_tuple=(keys, kg))

        else:
            if num_processes == None:
                num_processes = multiprocessing.cpu_count()

        chunk_keys = split_list_in_chunks(input_list=keys, num_chunks=num_processes)
        triples = []

        with Pool(num_processes) as p:
            triple_lists = p.map(self._extract, [(subset_keys, kg) for subset_keys in chunk_keys])

        for triple_list in triple_lists:
            triples += triple_list

        return triples


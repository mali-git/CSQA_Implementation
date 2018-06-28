import pickle
from collections import OrderedDict
import numpy as np
import json
from gensim.models import Word2Vec

def create_test_resources_dialogue_instance_creator():
    csqa_vocab_context_frq = OrderedDict()
    csqa_vocab_context_frq['cristiano'] = 3
    csqa_vocab_context_frq['ronaldo'] = 3
    csqa_vocab_context_frq['final'] = 10
    csqa_vocab_context_frq['june'] = 10
    csqa_vocab_context_frq['soccer'] = 1
    csqa_vocab_context_frq['goal'] = 10

    out = '../test_resources/csqa_vocab_context_vocab_5.pkl'

    with open(out, 'wb') as handle:
        pickle.dump(csqa_vocab_context_frq, handle, protocol=pickle.HIGHEST_PROTOCOL)

    csqa_vocab_response_frq = OrderedDict()
    csqa_vocab_response_frq['lionel'] = 1
    csqa_vocab_response_frq['messi'] = 1
    csqa_vocab_response_frq['argentine'] = 1
    csqa_vocab_response_frq['aguero'] = 1
    csqa_vocab_response_frq['striker'] = 1
    csqa_vocab_response_frq['draw'] = 1

    out = '../test_resources/csqa_vocab_response_vocab_5.pkl'

    with open(out, 'wb') as handle:
        pickle.dump(csqa_vocab_response_frq, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sent_1 = 'cristiano ronaldo final june soccer goal'.split()
    sent_2 = 'lionel messi argentine aguero striker draw'.split()
    sentences = [sent_1, sent_2]

    out = '../test_resources/test_soccer_word_to_vec'
    model = Word2Vec(sentences, size=100, window=1, min_count=1, workers=4)

    model.save(out)

    out = '../test_resources/kg_entity_to_embeddings.pkl'
    kg_entity_to_embeddings = OrderedDict()
    kg_entity_to_embeddings['Q11571'] = np.ones(shape=(100,))
    kg_entity_to_embeddings['Q615'] = np.ones(shape=(100,)) + 1.
    kg_entity_to_embeddings['Q2291862'] = np.ones(shape=(100,)) + 2.

    with open(out, 'wb') as handle:
        pickle.dump(kg_entity_to_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out = '../test_resources/kg_relations_to_embeddings.pkl'
    kg_relation_to_embeddings = OrderedDict()
    kg_relation_to_embeddings['P166'] = np.ones(shape=(100,)) + 2.
    kg_relation_to_embeddings['P19'] = np.ones(shape=(100,)) + 3.

    with open(out, 'wb') as handle:
        pickle.dump(kg_relation_to_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


    out = '../test_resources/filtered_entity_mapping.json'
    entity_mapping = OrderedDict()
    entity_mapping['Q11571'] = 'cristiano ronaldo'
    entity_mapping['Q615'] = 'lionel messi'

    with open(out, 'w') as outfile:
        json.dump(entity_mapping, outfile)






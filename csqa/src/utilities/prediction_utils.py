import numpy as np

from utilities.constants import KG_WORD


def replace_kg_tokens_with_predicted_entities(predicted_toks, predicted_entities, pred_prob_entities, kg_tok,
                                              kg_id_to_kg_entity, response_tok_id_to_word):
    predicted_entities = np.array(predicted_entities, dtype=np.int32).flatten()
    pred_prob_entities = np.array(pred_prob_entities, dtype=np.float32)
    descending_order_indices = np.argsort(pred_prob_entities)[::-1]
    predicted_entities = predicted_entities[descending_order_indices]

    indices = np.where(predicted_toks == kg_tok)

    str_pred_utter = np.vectorize(response_tok_id_to_word.get)(predicted_toks).tolist()

    if len(indices[0]) == 0:
        return str_pred_utter, []
    elif len(indices[0]) >= 1 and len(indices[0]) > predicted_entities.size:
        diff = len(indices[0]) - predicted_entities.size
        predicted_entities = np.concatenate([predicted_entities, predicted_entities[0:diff]], axis=-1)
    elif len(indices[0]) >= 1 and len(indices[0]) < predicted_entities.size:
        predicted_entities = predicted_entities[:len(indices[0])]

    predicted_entities = np.vectorize(kg_id_to_kg_entity.get)(predicted_entities)

    predicted_entities = np.array(predicted_entities, dtype=np.str)

    for i, index in enumerate(indices[0]):
        str_pred_utter[index] = predicted_entities[i]

    return str_pred_utter, predicted_entities


if __name__ == '__main__':
    entity_probs = [0.2, 0.8]
    entities = [[1], [2]]
    predicted_toks = np.array([4, 9, 9, 9, 10, 10, 4, 10, 10, 10, 10], dtype=np.int32)
    print("Length of predicted tokens: ", len(predicted_toks))

    id_to_kg_entity = {1: 'cristiano ronaldo', 2: 'Lionel Messi'}
    id_to_word = {4: KG_WORD, 7: 'Christiano Ronaldo', 8: 'Lionel Messi', 9: 'Aguero', 10: 'goal'}
    entiy_integrated = replace_kg_tokens_with_predicted_entities(predicted_toks=predicted_toks,
                                                                 predicted_entities=entities,
                                                                 pred_prob_entities=entity_probs,
                                                                 kg_tok=4,
                                                                 kg_id_to_kg_entity=id_to_kg_entity,
                                                                 response_tok_id_to_word=id_to_word)
    print(entiy_integrated)


import numpy as np

def replace_kg_tokens_with_predicted_entities(predicted_tok_ids, predicted_entities, pred_prob_entities, kg_tok_id):
    predicted_tok_ids = np.array(predicted_tok_ids,dtype=np.int32)
    predicted_entities = np.array(predicted_entities,dtype=np.int32).flatten()
    pred_prob_entities = np.array(pred_prob_entities,dtype=np.float32)
    descending_order_indices = np.argsort(pred_prob_entities)[::-1]
    predicted_entities = predicted_entities[descending_order_indices]

    indices = np.where(predicted_tok_ids==kg_tok_id)

    if len(indices[0]) == 0:
        return predicted_tok_ids
    elif len(indices[0]) >= 1 and len(indices[0]) >predicted_entities.size:
        diff = len(indices[0]) - predicted_entities.size
        predicted_entities = np.concatenate([predicted_entities,predicted_entities[0:diff]],axis=-1)
    elif len(indices[0]) >=1 and len(indices[0])<predicted_entities.size:
        predicted_entities = predicted_entities[:len(indices[0])]

    print("Copy: ",indices[0])
    print("predicted_entities: ", predicted_entities)

    np.put(predicted_tok_ids, indices[0],predicted_entities)

    return predicted_tok_ids.tolist()

if __name__ == '__main__':
    entity_probs = [1.]
    entities = [[1]]
    predicted_toks = [4,7,8,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    print("Length of predicted tokens: ",len(predicted_toks))

    entiy_integrated = replace_kg_tokens_with_predicted_entities(predicted_tok_ids=predicted_toks,
                                                                 predicted_entities=entities,
                                                                 pred_prob_entities=entity_probs,
                                                                 kg_tok_id=4)
    print(predicted_toks)
    print(entiy_integrated)
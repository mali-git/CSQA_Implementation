import logging

import bisect
import numpy as np
import spacy
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
nlp_parser = spacy.load('en')


def get_offsets_of_entity_with_longer_span(new_start, new_end, existing_start, existing_end):
    """
    Returns offsets of entity with bigger span
    :param new_start: Start position of new entity
    :param new_end: End position of new entity
    :param existing_start: Start position of existing entity
    :param existing_end: End position of existing entity
    :rtype: int,int
    """
    if (new_end - new_start) > (existing_end - existing_start):
        return new_start, new_end
    else:
        return existing_start, existing_end


def save_insertion_of_offsets(start_offsets, end_offsets, new_start, new_end):
    """
    Will safely add new offsets i.e. overlapping cases of entities will be solved.
    :param start_offsets: Start positions of current entities
    :param end_offsets: End positions of current entities
    :param new_start: Start position of new entity
    :param new_end: End position of new entity
    :rtype: list,list
    """
    start_offsets = np.array(start_offsets, dtype=int)
    end_offsets = np.array(end_offsets, dtype=int)
    indices = np.ones(shape=len(start_offsets))

    # Case 1: New entity is contained in existing entity. ['Chancellor of Germany'] Insertion Request: 'Germany'
    case_one_overlappings = (np.all([(start_offsets <= new_start), (end_offsets >= new_start)], axis=0) * 1).nonzero()
    # Case 2: New entity contains other entities. ['Germany'] Insertion Request: 'Chancellor of Germany'
    case_two_overlappings = (np.all([(start_offsets >= new_start), (end_offsets <= new_end)], axis=0) * 1).nonzero()

    # Case: New entity overlaps with more than one entity. Consider this as an error, therefore don't consider entity.
    # In next version deal this case.
    # TODO: In version 0.1.2 handle this case
    if len(case_one_overlappings[0]) != 0 and (len(case_two_overlappings[0]) != 0):
        return start_offsets, end_offsets

    mask = case_one_overlappings if (len(case_one_overlappings[0]) == 1) else case_two_overlappings

    # Entity doesn't overlap with existing entities -> Insert offsets
    if len(mask[0]) == 0:
        start_offsets = start_offsets.tolist()
        end_offsets = end_offsets.tolist()
        bisect.insort_left(start_offsets, new_start)
        bisect.insort_left(end_offsets, new_end)
        return start_offsets, end_offsets

    problematic_start, problematic_end = start_offsets[mask][0], end_offsets[mask][0]
    valid_start, valid_end = get_offsets_of_entity_with_longer_span(new_start=new_start, new_end=new_end,
                                                                    existing_start=problematic_start,
                                                                    existing_end=problematic_end)

    start_offsets = start_offsets.tolist()
    end_offsets = end_offsets.tolist()
    index = mask[0][0]

    if valid_start == start_offsets[index] and valid_end == end_offsets[index]:
        log.info("Don't insert new entity since it is covered by existing entity")
    else:
        log.info("Conflict due to overlapping of entities")
        # Remove invalid offsets and add valid ones
        log.info("Remove offsets (start:%s,end:%s)" % (new_start, new_end))
        del start_offsets[index]
        del end_offsets[index]
        bisect.insort_left(start_offsets, valid_start)
        bisect.insort_left(end_offsets, valid_end)

    return start_offsets, end_offsets


def mark_parts_in_text(start_offsets_entities, end_offsets_entities, text):
    """
    Mark parts which refer to entity with True and non-entity parts with False
    :param start_offsets_entities: Start positions of relevant entities in text.
    :param end_offsets_entities: End positions of relevant entities in text.
    :param text: Text to analyse
    :rtype: dict
    """
    offsets_info_dict = OrderedDict()
    current_pos_in_utterance = 0

    for i in range(len(start_offsets_entities)):
        start = current_pos_in_utterance
        end = start_offsets_entities[i]
        # Start of new token begins after the last character of the entity
        current_pos_in_utterance = end_offsets_entities[i]

        # Add part info
        if end - start <= 0:
            # Start of current part in utterance is the beginning of an entity -> Add only entity info
            offsets_info_dict[(start_offsets_entities[i], end_offsets_entities[i])] = True
        else:
            # Part doesn't refer to an entity
            offsets_info_dict[(start, end)] = False
            # Part does refer to an entity
            offsets_info_dict[(start_offsets_entities[i], end_offsets_entities[i])] = True

    if not start_offsets_entities:
        return offsets_info_dict

    end_of_last_entity = end_offsets_entities[-1]

    text_length = len(text)

    if text_length - end_of_last_entity > 0:
        offsets_info_dict[(end_of_last_entity, text_length)] = False

    return offsets_info_dict


def compute_nlp_features(txt, offsets_info_dict):
    """
    Computes all different NLP features (tokens,part-of-speech tags, dependency parsing features)
    :param txt: Text for which NLP features should be computed
    :param offsets_info_dict: Dictionary with marked parts of txt. Indicates whether part contains relevant entity
    or not
    :rtype: list
    """
    doc = nlp_parser(u'%s' % (txt))
    spans = []

    for offset_tuple, is_entity in offsets_info_dict.items():
        start = offset_tuple[0]
        end = offset_tuple[1]

        if not is_entity:
            # Parts not corresponding to an entity can include spaces before the first token and after the last token.
            # Remove space, so that valid spacy span can be created
            if txt[start:start+1].isspace():
                start += 1
            if end != len(txt):
                end -= 1

        span = doc.char_span(start, end, label=int(is_entity))
        spans.append(span)
    return spans

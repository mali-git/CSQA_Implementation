import logging

import bisect
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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

    # Case 1: New entity is contained in existing entity. ['Chancellor of Germany'] Insert: 'Germany'
    case_one_overlappings = (np.all([(start_offsets <= new_start), (end_offsets >= new_start)], axis=0) * 1).nonzero()
    # Case 2: New entity contains other entities. ['Germany'] Insert: 'Chancellor of Germany'
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
        log.info("Remove start offsets (start:%s,end:%s)" % (new_start, new_end))
        del start_offsets[index]
        del end_offsets[index]
        bisect.insort_left(start_offsets, valid_start)
        bisect.insort_left(end_offsets, valid_end)

    return start_offsets, end_offsets

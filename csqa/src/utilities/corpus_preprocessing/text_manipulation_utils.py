import logging

import bisect
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def handle_invalid_case(start, end, problematic_start, problematic_end):
    if (end - start) > (problematic_end - problematic_start):
        return start, end
    else:
        return problematic_start, problematic_end


def save_insertion_of_offsets(start_offsets, end_offsets, start, end):
    start_offsets = np.array(start_offsets, dtype=int)
    end_offsets = np.array(end_offsets, dtype=int)
    indices = np.ones(shape=len(start_offsets))

    # Case 1: New entity is contained in existing entity
    case_one_invalid = (((start_offsets <= start) and (end_offsets >= start)) * 1).nonzero()
    # Case 2: New entity contains other entities
    case_two_invalid = (((start_offsets >= start) and (end_offsets <= end)) * 1).nonzero()

    mask = case_one_invalid if (len(case_one_invalid[0]) == 1) else case_two_invalid

    problematic_start, problematic_end = start_offsets[mask][0], end_offsets[mask][0]
    valid_start, valid_end = handle_invalid_case(start=start, end=end, problematic_start=problematic_start,
                                                 problematic_end=problematic_end)

    start_offsets = start_offsets.tolist()
    end_offsets = end_offsets.tolist()
    index = mask[0][0]

    if valid_start==start_offsets[index] and valid_end==end_offsets[index]:
        log.info("Don't insert new entity since it is covered by exsting entity")
    else:
        log.info("Conflict due to overlapping of entities")
        # Remove invalid offsets and add valid ones
        log.info("Remove start offsets (start:%s,end:%s)" % (start,end))
        del start_offsets[index]
        del end_offsets[index]
        bisect.insort_left(start_offsets, valid_start)
        bisect.insort_left(end_offsets, valid_end)

    return start_offsets, end_offsets



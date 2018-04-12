import logging

import bisect
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_insertion_of_offsets(start_offsets, end_offsets, start, end):
    # Get the smallest value where start can be inserted in the sorted list
    index_of_smallest = bisect.bisect(start_offsets, start)
    indices_of_overlapped_offsets = []
    save_start_offsets = []
    save_end_offsets = []

    if index_of_smallest == 0:
        log.info(
            "start is not in start offsets --> All existing end offsets smaller than current end are overlapped")
        # start is not in start offsets --> All existing end offsets smaller than current end are overlapped
        temp_np_array = np.array(end_offsets, dtype=int)
        current_indices_of_none_overlapped_offsets = ((temp_np_array > end) * 1).nonzero()
        save_start_offsets = start_offsets[current_indices_of_none_overlapped_offsets]
        save_end_offsets = end_offsets[current_indices_of_none_overlapped_offsets]


    elif index_of_smallest == len(start_offsets):
        # start is greater then all current values --> there are no overlapping offsets
        save_start_offsets = start_offsets
        save_end_offsets = end_offsets

    else:
        # start lies between smallest and biggest current start

        # Step 1: Get indices of end positions which are bigger or equal to current end
        temp_np_array = np.array(end_offsets, dtype=int)
        current_end_offsets_indices_bigger_or_equal = ((temp_np_array >= end) * 1).nonzero()
        non_over_lapping_offsets = np.ones(shape=len(start_offsets)).tolist()
        insertion_allowed = False

        # Step 2: Check relevant start indices
        # relevant_start_indices = start_offsets[current_end_offsets_indices_bigger_or_equal]
        for relevant_index in current_end_offsets_indices_bigger_or_equal:
            current_start = start_offsets[relevant_index]
            if current_start <= start:
                # Case: ['Chancellor of Germany'] Insert: 'Germany' -> Don't insert
                continue
            else:
                # Case: ['Germany'] Insert: 'Chancellor of Germany' -> Insert
                non_over_lapping_offsets[relevant_index] = 0
                insertion_allowed = True

        # Step 3: Determine safe start offsets
        save_start_offsets = np.array(start_offsets[non_over_lapping_offsets], dtype=int).nonzero().tolist()
        # Insert new start if possible
        if insertion_allowed:
            bisect.insort_left(save_start_offsets, start)
            bisect.insort_left(save_end_offsets, end)

    return save_start_offsets, save_end_offsets

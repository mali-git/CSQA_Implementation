import logging
import unittest

from utilities.corpus_preprocessing.text_manipulation_utils import save_insertion_of_offsets, mark_parts_in_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestUtterance2TensorCreator(unittest.TestCase):
    def test_save_insertion_of_offsets(self):
        start_offsets = [10]
        end_offsets = [40]

        # Case: ['Chancellor of Germany'] Insert: 'Germany' --> Don't insert
        start_new = 30
        end_new = 40

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start_new,
                                                                               end_new)

        self.assertTrue(len(updated_start_offsets) == len(updated_end_offsets) == 1)
        self.assertEqual(updated_start_offsets[0], start_offsets[0])
        self.assertEqual(updated_end_offsets[0], end_offsets[0])

        # Case: ['Germany'] Insert: 'Chancellor of Germany' --> Insert
        start_new = 5
        end_new = 40

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start_new,
                                                                               end_new)
        self.assertTrue(len(updated_start_offsets) == len(updated_end_offsets) == 1)
        self.assertEqual(updated_start_offsets[0], start_new)
        self.assertEqual(updated_end_offsets[0], end_new)

        # Case: Add non-overlapping entity
        start_new = 5
        end_new = 20

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start_new,
                                                                               end_new)

        self.assertTrue(len(updated_start_offsets) == len(updated_end_offsets) == 2)
        self.assertEqual(updated_start_offsets[0], start_new)
        self.assertEqual(updated_end_offsets[0], end_new)
        self.assertEqual(updated_start_offsets[1], start_offsets[0])
        self.assertEqual(updated_end_offsets[1], end_offsets[0])

        # Case: Entity overlaps with more than one entity -> Don't insert new entity
        start_offsets = [10, 25]
        end_offsets = [20, 35]
        start_new = 15
        end_new = 25

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start_new,
                                                                               end_new)

        self.assertTrue(len(updated_start_offsets) == len(updated_end_offsets) == 2)
        self.assertEqual(updated_start_offsets[0], start_offsets[0])
        self.assertEqual(updated_end_offsets[0], end_offsets[0])
        self.assertEqual(updated_start_offsets[1], start_offsets[1])
        self.assertEqual(updated_end_offsets[1], end_offsets[1])

    def test_mark_parts_in_text(self):
        text = 'Where in Berlin does the chancellor of Germany live?'

        start_entity_one = text.find('Berlin')
        end_entity_one = text.find('Berlin') + len('Berlin')

        start_entity_two = text.find('chancellor of Germany')
        end_entity_two = start_entity_two + len('chancellor of Germany')
        start_offsets = [start_entity_one, start_entity_two]
        end_offsets = [end_entity_one, end_entity_two]

        offsets_info_dict = mark_parts_in_text(start_offsets_entities=start_offsets, end_offsets_entities=end_offsets,
                                               text=text)

        correct_entity_info = [False, True, False, True, False]
        correct_parts = [(0, 9), (9, 15), (15, 25), (25, 46), (46, 52)]
        counter = 0

        self.assertEqual(len(offsets_info_dict), 5)

        for key, value in offsets_info_dict.items():
            self.assertEqual(value, correct_entity_info[counter])
            self.assertEqual(key, correct_parts[counter])
            counter += 1

        # Case: Last word is entity, and afterwards there is a punctuation
        text = 'How big is the population of Germany?'
        start_entity = text.find('Germany')
        end_entity = start_entity + len('Germany')

        start_offsets = [start_entity]
        end_offsets = [end_entity]

        offsets_info_dict = mark_parts_in_text(start_offsets_entities=start_offsets, end_offsets_entities=end_offsets,
                                               text=text)

        correct_entity_info = [False, True, False]
        correct_parts = [(0, start_entity), (start_entity, end_entity), (end_entity, end_entity + 1)]
        print(correct_parts)
        counter = 0

        self.assertEqual(len(offsets_info_dict), 3)

        for key, value in offsets_info_dict.items():
            self.assertEqual(value, correct_entity_info[counter])
            self.assertEqual(key, correct_parts[counter])
            counter += 1

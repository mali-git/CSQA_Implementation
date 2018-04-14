import logging
import unittest

from utilities.corpus_preprocessing.text_manipulation_utils import save_insertion_of_offsets

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
        # print(updated_start_offsets)

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



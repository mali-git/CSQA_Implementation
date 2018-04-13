
import unittest
import logging

from utilities.corpus_preprocessing.text_manipulation_utils import save_insertion_of_offsets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TestUtterance2TensorCreator(unittest.TestCase):

    def test_save_insertion_of_offsets(self):
        start_offsets = [10]
        end_offsets = [40]

        # Case: ['Chancellor of Germany'] Insert: 'Germany' --> Don't insert
        start = 30
        end = 40

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start, end)
        # print(updated_start_offsets)

        self.assertTrue(len(updated_start_offsets)==len(updated_end_offsets)==1)
        self.assertEqual(updated_start_offsets[0],start_offsets[0])
        self.assertEqual(updated_end_offsets[0],end_offsets[0])

        # Case: ['Germany'] Insert: 'Chancellor of Germany' --> Insert
        start = 5
        end = 40

        updated_start_offsets, updated_end_offsets = save_insertion_of_offsets(start_offsets, end_offsets, start, end)
        print(updated_start_offsets)
        self.assertTrue(len(updated_start_offsets) == len(updated_end_offsets) == 1)


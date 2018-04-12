
import unittest
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TestUtterance2TensorCreator(unittest.TestCase):

    def test_save_insertion_of_offsets(self):
        start_offsets = [10]
        end_offsets = [40]

        # Case: ['Chancellor of Germany'] Insert: 'Germany' --> Don't insert
        start = 30
        end = 40


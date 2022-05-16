import data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        i_i, a_m = data_utils.encode_data(self.dataset,
                                          self.tokenizer,
                                          self.max_seq_len)
        self.assertEqual(i_i.shape, torch.Size([len(self.dataset),
                                                    self.max_seq_len]))
        self.assertEqual(i_i.dtype, torch.long)
        self.assertEqual(a_m.shape, torch.Size([len(self.dataset),
                                                    self.max_seq_len]))
        self.assertEqual(a_m.dtype, torch.long)

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        l = data_utils.extract_labels(self.dataset)
        self.assertEqual(set(l), {1,0})

if __name__ == "__main__":
    unittest.main()

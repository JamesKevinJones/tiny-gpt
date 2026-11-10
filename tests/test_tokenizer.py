import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tokenizer import PAT, Tokenizer, train_bpe


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample = "Once upon a time, a cat met a dog. Héllo 123! " * 30
        cls.tokenizer = Tokenizer(train_bpe(Counter(PAT.findall(sample)), 300))

    def test_round_trip_keeps_unicode_punctuation_and_spaces(self):
        text = "Héllo, cat!  123\nA dog."
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_merge_order_builds_a_token_from_prior_merges(self):
        tokenizer = Tokenizer([(ord("a"), ord("b")), (256, ord("c"))])
        self.assertEqual(tokenizer.encode("abc"), [257])
        self.assertEqual(tokenizer.decode([257]), "abc")

    def test_end_of_text_does_not_appear_in_decoded_story(self):
        ids = self.tokenizer.encode("One.")
        self.assertEqual(self.tokenizer.decode(ids + [self.tokenizer.eot]), "One.")

    def test_saved_tokenizer_has_the_same_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            restored = Tokenizer.load(path)
            text = "A cat met a dog."
            self.assertEqual(restored.encode(text), self.tokenizer.encode(text))
            self.assertEqual(restored.decode(restored.encode(text)), text)


if __name__ == "__main__":
    unittest.main()

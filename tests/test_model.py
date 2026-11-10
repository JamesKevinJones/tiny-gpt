import unittest

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(vocab_size=16, block_size=4, n_layer=2, n_head=2, n_embd=16))
        self.model.eval()

    def test_future_token_cannot_change_prior_logits(self):
        with torch.no_grad():
            first, _ = self.model(torch.tensor([[1, 2, 3, 4]]))
            second, _ = self.model(torch.tensor([[1, 2, 8, 9]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_training_loss_uses_all_target_positions(self):
        inputs = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        logits, loss = self.model(inputs, targets)
        expected = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        torch.testing.assert_close(loss, expected)

    def test_output_projection_shares_token_embedding_weights(self):
        self.assertEqual(self.model.lm_head.weight.data_ptr(), self.model.wte.weight.data_ptr())
        optimizer = self.model.configure_optimizer(0.001, 0.1, "cpu")
        weight_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(weight_ids.count(id(self.model.wte.weight)), 1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0)

    def test_generation_truncates_the_context_window(self):
        prompt = torch.tensor([[1, 2, 3, 4]])
        generated = self.model.generate(prompt, max_new_tokens=3, top_k=1)
        self.assertEqual(generated.shape, (1, 7))
        torch.testing.assert_close(generated[:, :4], prompt)


if __name__ == "__main__":
    unittest.main()

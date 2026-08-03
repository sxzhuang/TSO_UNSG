import random
import unittest
from pathlib import Path

import torch

from env import load_game_pickle
from network import DefenderTreeSampler, EvaderTreeSampler
from trainer import TSOTrainer, conditional_pruned_probability


class TSOCoreTests(unittest.TestCase):
    def setUp(self):
        random.seed(3407)
        torch.manual_seed(3407)
        self.device = torch.device("cpu")
        self.game = load_game_pickle(Path("env/s1.pkl"), enumerate_actions=True)
        self.evader = EvaderTreeSampler(self.game, self.device)
        self.defender = DefenderTreeSampler(self.game, self.device)

    def test_s1_uses_92_by_110_legal_action_space(self):
        self.assertEqual(len(self.game.evader_actions), 92)
        self.assertEqual(len(self.game.defender_actions), 110)
        self.assertTrue(all(action[0] != action[1] for action in self.game.defender_actions))
        self.assertEqual(self.game.payoff_matrix.shape, (92, 110))

    def test_pruned_probability_is_conditionally_normalized(self):
        raw_alternative = torch.tensor([0.3, 0.1])
        original = torch.tensor([0.6, 0.6])
        normalized = conditional_pruned_probability(raw_alternative, original)
        self.assertTrue(torch.allclose(normalized, torch.tensor([0.75, 0.25])))
        self.assertFalse(normalized.requires_grad)

    def test_tree_probabilities_form_distributions(self):
        evader_total = sum(self.evader.action_probability(action) for action in self.game.evader_actions)
        defender_total = sum(self.defender.action_probability(action) for action in self.game.defender_actions)
        self.assertAlmostEqual(evader_total, 1.0, places=5)
        self.assertAlmostEqual(defender_total, 1.0, places=5)

    def test_sample_and_prune_excludes_exact_ordered_action(self):
        with torch.no_grad():
            original, _, original_q = self.defender.sample_batch_paths(
                8, sampling_epsilon=None, behavior_epsilon=0.8
            )
        alternative, policy_probability, raw_q = self.defender.sample_batch_paths(
            8,
            sampling_epsilon=0.8,
            behavior_epsilon=0.8,
            excluded_paths=original,
        )
        self.assertTrue(all(first != second for first, second in zip(original, alternative)))
        self.assertTrue(policy_probability.requires_grad)
        self.assertFalse(raw_q.requires_grad)
        actual_p = conditional_pruned_probability(raw_q, original_q)
        self.assertTrue(torch.all(actual_p > 0))
        self.assertTrue(torch.all(actual_p <= 1.0 + 1e-6))

    def test_training_estimator_only_differentiates_policy_probability(self):
        trainer = TSOTrainer(
            self.evader,
            self.defender,
            self.game,
            epsilon=0.8,
            tau=0.05,
            evader_lr=1e-4,
            defender_lr=1e-4,
            tau_decay=0.7,
            evader_lr_decay=0.8,
            defender_lr_decay=0.8,
        )
        result = trainer.train_batch(batch_size=3)
        self.assertTrue(result.evader_loss.requires_grad)
        self.assertTrue(result.defender_loss.requires_grad)
        self.assertFalse(result.evader_sample_probability.requires_grad)
        self.assertFalse(result.defender_sample_probability.requires_grad)
        result.evader_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.evader.parameters()))
        result.defender_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.defender.parameters()))


if __name__ == "__main__":
    unittest.main()

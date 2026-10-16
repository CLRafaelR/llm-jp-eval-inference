import copy
import json
import unittest

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from inference import VLLMGenerator
from structured_decisions import JointSchemaDecisionEncoder


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class StructuredDecisionTests(unittest.TestCase):
    def setUp(self):
        self.generator = VLLMGenerator.__new__(VLLMGenerator)
        self.generator.decision_encoder = JointSchemaDecisionEncoder(CharacterTokenizer(), 16384)
        self.generator.model = SimpleNamespace(encode=Mock())
        self.target_data = {
            "target_dataset": "test",
            "instruction": "正しい選択肢を選んでください。",
            "label_list": ["C", "A", "B"],
            "samples": [{"input": "問題1"}, {"input": "問題2"}],
        }

    def set_scores(self, *scores):
        self.generator.model.encode.return_value = [
            SimpleNamespace(outputs=SimpleNamespace(data=torch.tensor(row, dtype=torch.float64))) for row in scores
        ]

    def test_answers_follow_encoded_option_order_and_preserve_output_format(self):
        self.set_scores([0.1, 0.7, 0.2], [0.4, 0.4, 0.2])
        original = copy.deepcopy(self.target_data)

        results = self.generator._generate_structured_decisions(self.target_data)

        expected_answers = [
            {"type": "choice", "choice": "B", "probabilities": {"A": 0.1, "B": 0.7, "C": 0.2}, "confidence": 0.7},
            {"type": "choice", "choice": "A", "probabilities": {"A": 0.4, "B": 0.4, "C": 0.2}, "confidence": 0.4},
        ]
        for sample, answer in zip(results["samples"], expected_answers, strict=True):
            self.assertEqual(sample["structured_decision"], {"answer": answer})
            self.assertEqual(json.loads(sample["generated"]), {"answers": {"answer": answer}})
        self.assertEqual(self.target_data, original)
        self.assertEqual(self.generator.model.encode.call_args.kwargs["pooling_task"], "token_classify")

    def test_rejects_incorrect_output_count(self):
        self.set_scores([0.1, 0.7, 0.2])
        with self.assertRaisesRegex(RuntimeError, "Expected 2 decision outputs, got 1"):
            self.generator._generate_structured_decisions(self.target_data)

    def test_rejects_incorrect_probability_count(self):
        for scores in ([0.1, 0.9], [0.1, 0.2, 0.3, 0.4]):
            with self.subTest(scores=scores):
                self.set_scores(scores, [0.1, 0.7, 0.2])
                with self.assertRaisesRegex(RuntimeError, "Expected 3 option probabilities"):
                    self.generator._generate_structured_decisions(self.target_data)

    def test_rejects_missing_or_duplicate_labels_before_inference(self):
        for labels, message in (([], "no label_list"), ([1, "1"], "duplicate labels")):
            with self.subTest(labels=labels):
                self.target_data["label_list"] = labels
                with self.assertRaisesRegex(ValueError, message):
                    self.generator._generate_structured_decisions(self.target_data)
        self.generator.model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()

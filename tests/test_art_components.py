from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from sovereign_ai.art_field import ARTField
from sovereign_ai.associative_coupling import AssociativeProjection
from sovereign_ai.evaluation import ARTValueField
from sovereign_ai.tracing import TraceRecorder
from sovereign_ai.vigilance import VigilanceController


class ARTFieldTests(unittest.TestCase):
    def test_category_creation_resonance_reset_and_capacity(self) -> None:
        field = ARTField(input_dim=3, max_categories=2, vigilance=0.95)
        first = field.process(np.array([1.0, 0.8, 0.0]))
        self.assertTrue(first.resonance)
        self.assertEqual(len(field.categories), 1)
        self.assertIs(field.last_result, first)

        match = field.process(np.array([1.0, 0.75, 0.0]))
        self.assertTrue(match.resonance)
        self.assertEqual(match.search_path, [0])

        mismatch = field.process(np.array([0.0, 0.0, 1.0]))
        self.assertTrue(mismatch.resonance)
        self.assertGreater(len(mismatch.search_path), 1)
        self.assertEqual(len(field.categories), 2)

        capped = field.process(np.array([0.0, 1.0, 0.0]))
        self.assertLessEqual(len(field.categories), 2)
        self.assertIsNotNone(capped.last_result if hasattr(capped, "last_result") else field.last_result)


class AssociativeProjectionTests(unittest.TestCase):
    def test_projection_shape_learning_resizing_and_empty_result(self) -> None:
        source = ARTField(input_dim=2, max_categories=3)
        target = ARTField(input_dim=2, max_categories=3)
        projection = AssociativeProjection(source, target, "source->target")
        self.assertEqual(projection.project().shape, (0,))

        source_result = source.process(np.array([1.0, 0.0]))
        target_result = target.process(np.array([0.0, 1.0]))
        projection.learn(source_result.category_activation, target_result.category_activation)
        self.assertEqual(projection.weights.shape, (1, 1))
        self.assertGreater(np.linalg.norm(projection.project(source_result.category_activation)), 0.0)

        source.process(np.array([0.0, 1.0]))
        target.process(np.array([1.0, 0.0]))
        projection.project()
        self.assertEqual(projection.weights.shape, (2, 2))

        empty_source = ARTField(input_dim=2)
        empty_projection = AssociativeProjection(empty_source, target, "empty->target")
        self.assertTrue(np.allclose(empty_projection.project(), np.zeros(len(target.categories))))


class VigilanceControllerTests(unittest.TestCase):
    def test_bounded_independent_learning(self) -> None:
        controller = VigilanceController(["perception", "action"])
        activation = np.array([1.0, 0.0])
        before_perception = controller.modulation("perception", activation, mismatch=1.0, reward=-1.0)
        controller.learn("perception", activation, mismatch=1.0, reward=-1.0)
        after_perception = controller.modulation("perception", activation, mismatch=1.0, reward=-1.0)
        action_value = controller.modulation("action", activation, mismatch=1.0, reward=-1.0)
        self.assertGreater(after_perception, before_perception)
        self.assertNotEqual(after_perception, action_value)
        self.assertLessEqual(abs(after_perception), 0.2)

        before_reward = controller.modulation("perception", activation, mismatch=0.0, reward=1.0)
        controller.learn("perception", activation, mismatch=0.0, reward=1.0)
        after_reward = controller.modulation("perception", activation, mismatch=0.0, reward=1.0)
        self.assertLessEqual(after_reward, before_reward)


class TraceRecorderTests(unittest.TestCase):
    def test_json_and_summary(self) -> None:
        recorder = TraceRecorder()
        recorder.record_field("perception", 0, 0, 1, True, 0.8, 0.9, [0, 1], 0.1, 0.2, 3)
        recorder.record_projection("p->v", 0, 0.5, 2, 4)
        recorder.record_behavior(0, 2, 1.0, 0.4, 0.7, 0.2, True)
        summary = recorder.summary()
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["fields"]["perception"]["categories"], 3)
        self.assertEqual(summary["action_distribution"], {"2": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            recorder.to_json(path)
            self.assertTrue(path.exists())


class ValueFieldIndexTests(unittest.TestCase):
    def test_no_fixed_semantic_categories(self) -> None:
        field = ARTValueField(max_perceptual_categories=3)
        self.assertEqual(len(field.categories), 0)
        first = field.resonate_value(np.array([1.0, 0.0, 0.0]), 1.0, 0.0, 0.0, 0.0, np.array([1.0]), learn=True)
        second = field.resonate_value(np.array([0.0, 1.0, 0.0]), -1.0, 0.9, 0.0, 0.0, np.array([1.0]), learn=True)
        self.assertGreaterEqual(len(field.categories), 1)
        self.assertEqual(field.category_values.shape[0], len(field.categories))
        self.assertNotEqual(first.result.prediction_error, second.result.prediction_error)


if __name__ == "__main__":
    unittest.main()

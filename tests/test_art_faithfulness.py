"""Faithfulness tests covering learned behavior across ART components."""

from __future__ import annotations

import unittest

import numpy as np

from sovereign_ai.action_selection import ARTActionField, BasalGangliaActionSelection
from sovereign_ai.architecture import CognitiveArchitecture
from sovereign_ai.associative_coupling import AssociativeProjection
from sovereign_ai.environment import GridWorld
from sovereign_ai.evaluation import ARTValueField, ValueSystem
from sovereign_ai.goal_system import GoalSystem
from sovereign_ai.imagination import ARTExpectationField, ImaginationLoop
from sovereign_ai.perception import ARTPerception, ARTPerceptualField
from sovereign_ai.tracing import TraceRecorder
from sovereign_ai.transition_model import ARTTemporalField, LinearTransitionModel
from sovereign_ai.vigilance import VigilanceController


class ARTFaithfulnessTests(unittest.TestCase):
    """Validate the learned behavior of each coupled ART subsystem."""

    def test_value_categories_and_associations_are_learned(self) -> None:
        """Check that value categories and associations are learned."""

        field = ARTValueField(max_perceptual_categories=4, context_count=4)
        self.assertEqual(len(field.categories), 0)
        activation = np.array([1.0, 0.0, 0.0, 0.0])
        field.resonate_value(
            activation,
            reward=1.0,
            novelty=0.2,
            context=0.0,
            goal_alignment=0.5,
            action_distribution=np.array([1.0]),
            learn=True,
        )
        self.assertGreater(len(field.categories), 0)
        self.assertEqual(field.category_values.shape[0], len(field.categories))
        self.assertGreater(np.linalg.norm(field.category_values), 0.0)
        self.assertEqual(
            field.attentional_preferences.shape, (len(field.categories), 4)
        )

    def test_action_schemas_are_learned_from_experience(self) -> None:
        """Check that action schemas adapt from reward-driven experience."""

        field = ARTActionField(max_categories=4, action_count=3)
        self.assertEqual(len(field.categories), 0)
        percept = np.array([1.0, 0.0, 0.0, 0.0])
        result = field.select(percept, value=0.3, salience=np.array([0.0, 1.0, 0.0]))
        self.assertEqual(result.action_index, 1)
        self.assertGreater(len(field.categories), 0)
        before_count = len(field.categories)
        field.learn_action(percept, 2, reward_prediction_error=1.0)
        self.assertGreaterEqual(len(field.categories), before_count)
        self.assertGreater(np.max(field.action_associations[:, 2]), 0.0)

    def test_associative_projection_learns_coupling(self) -> None:
        """Check that projection weights change after a learning update."""

        source = ARTPerceptualField(input_dim=3, max_categories=3)
        target = ARTValueField(max_perceptual_categories=3, context_count=3)
        source_result = source.process(np.array([1.0, 0.0, 0.0]))
        target_state = target.resonate_value(
            source_result.category_activation,
            0.5,
            0.1,
            0.0,
            0.0,
            np.array([1.0]),
            learn=True,
        )
        projection = AssociativeProjection(source, target, "p->v")
        before = projection.project(source_result.category_activation)
        projection.learn(source_result.category_activation, target_state.activation)
        after = projection.project(source_result.category_activation)
        self.assertGreater(np.linalg.norm(after), np.linalg.norm(before))

    def test_imagination_rollout_uses_resonance_trace(self) -> None:
        """Check that imagination rollouts emit usable trace metadata."""

        perception = ARTPerceptualField(input_dim=3, max_categories=4, vigilance=0.4)
        perception.process(np.array([1.0, 0.0, 0.0]))
        value = ARTValueField(max_perceptual_categories=4, context_count=4)
        action = ARTActionField(max_categories=4, action_count=2)
        temporal = ARTTemporalField(
            state_dim=3, action_count=2, perceptual_category_count=4
        )
        temporal.learn(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        expectation = ARTExpectationField(
            perception, value, action.category_action_weights
        )
        rollout = expectation.prospective_rollout(
            np.array([1.0]),
            np.array([1.0]),
            np.empty(0),
            temporal,
            action,
            horizon=2,
        )
        self.assertGreaterEqual(len(rollout.trace), 1)
        self.assertIn("perception_resonance", rollout.trace[0])
        self.assertEqual(rollout.perceptual_bias.shape, (3,))
        self.assertIsInstance(rollout.value_category_bias, np.ndarray)
        self.assertTrue(any("accepted" in item for item in rollout.trace))

    def test_temporal_sequence_mismatch_sets_reset(self) -> None:
        """Check temporal mismatch tracking and repeated-learning behavior."""

        temporal = ARTTemporalField(
            state_dim=3, action_count=2, perceptual_category_count=3, max_categories=4
        )
        temporal.learn(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        prediction = temporal.predict_categories(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.empty(0),
        )
        self.assertGreaterEqual(prediction.mismatch, 0.0)
        self.assertGreaterEqual(len(temporal.mismatch_trace), 1)
        repeated_before_count = float(np.sum(temporal.sequence_counts))
        repeated_prediction = temporal.predict_categories(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.empty(0),
        )
        self.assertEqual(repeated_prediction.perceptual_bias.shape, (3,))
        for _ in range(3):
            temporal.learn(
                np.array([1.0, 0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        self.assertGreater(
            float(np.sum(temporal.sequence_counts)), repeated_before_count
        )

    def test_vigilance_controller_learns_adjustment(self) -> None:
        """Check that vigilance modulation changes after learning."""

        controller = VigilanceController(["perception"])
        activation = np.array([1.0, 0.0])
        before = controller.modulation(
            "perception", activation, mismatch=1.0, reward=-1.0
        )
        controller.learn("perception", activation, mismatch=1.0, reward=-1.0)
        after = controller.modulation(
            "perception", activation, mismatch=1.0, reward=-1.0
        )
        self.assertNotEqual(before, after)

    def test_compatibility_wrappers_preserve_constructors(self) -> None:
        """Check that compatibility aliases still construct the right classes."""

        self.assertIsInstance(ARTPerception(input_dim=3), ARTPerceptualField)
        self.assertIsInstance(ValueSystem(), ARTValueField)
        self.assertIsInstance(BasalGangliaActionSelection(4, 2), ARTActionField)
        self.assertIsInstance(GoalSystem(3), GoalSystem)
        self.assertIsInstance(LinearTransitionModel(3, 2), ARTTemporalField)
        perception = ARTPerceptualField(input_dim=3)
        value = ARTValueField(max_perceptual_categories=4)
        self.assertIsInstance(
            ImaginationLoop(perception, value, np.empty((0, 2))), ARTExpectationField
        )

    def test_integration_degrades_without_expectation_but_runs(self) -> None:
        """Run a short integration check over the coupled architecture."""

        env = GridWorld(size=5, seed=11)
        recorder = TraceRecorder()
        agent = CognitiveArchitecture(
            env.observation_dim,
            env.action_count,
            max_categories=8,
            seed=5,
            debug=False,
            trace_recorder=recorder,
        )
        initial_action_categories = len(agent.action_selector.categories)
        for step in range(10):
            agent.step(env, step)
        self.assertGreater(len(agent.perception.categories), 0)
        self.assertGreater(len(agent.value_system.categories), 0)
        self.assertGreater(
            len(agent.action_selector.categories), initial_action_categories
        )
        self.assertGreater(len(agent.transition_model.categories), 0)
        self.assertGreater(len(agent.transition_model.mismatch_trace), 0)
        self.assertTrue(
            any(
                field.last_result is not None
                for field in [
                    agent.perception,
                    agent.value_system,
                    agent.action_selector,
                    agent.transition_model,
                    agent.imagination,
                ]
            )
        )
        self.assertTrue(
            any(
                not trace.resonance or len(trace.search_path) > 1
                for trace in recorder.field_traces
            )
        )
        self.assertTrue(
            any(trace.update_norm > 0.0 for trace in recorder.projection_traces)
        )
        self.assertTrue(
            any(
                abs(value) > 0.0
                for weights in agent.vigilance_controller.weights.values()
                for value in weights
            )
        )
        summary = recorder.summary()
        self.assertGreater(summary["steps"], 0)
        self.assertIn("perception", summary["fields"])
        self.assertIn("reward_trend", summary)
        mismatches = agent.transition_model.mismatch_trace
        if len(mismatches) >= 4:
            midpoint = len(mismatches) // 2
            self.assertLessEqual(
                np.mean(mismatches[midpoint:]), np.mean(mismatches[:midpoint]) + 0.5
            )
        agent.imagination.categories = np.empty((0, agent.input_dim), dtype=float)
        agent.step(env, 11)


if __name__ == "__main__":
    unittest.main()

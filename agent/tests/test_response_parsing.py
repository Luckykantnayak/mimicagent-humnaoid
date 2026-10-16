"""Regressions for response envelopes observed in the Opus skill runs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tools import extract_json_from_message, extract_python_code_from_message
from agents.base_agent import BaseAgent


class ResponseParsingTest(unittest.TestCase):
    def test_streamed_usage_remains_available_to_pipeline(self):
        response = AIMessage(content='{}', usage_metadata={
            'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15})
        agent = BaseAgent(Mock(invoke=Mock(return_value=response)))
        self.assertEqual(agent.generate('prompt').response_metadata['usage']['output_tokens'], 3)

    def test_review_uses_answer_after_completed_reasoning(self):
        response = '<thinking>Candidate: {"accepted": true}</thinking>{"accepted": false, "issues": ["wrong axis"]}'
        self.assertEqual(extract_json_from_message(response),
                         {"accepted": False, "issues": ["wrong axis"]})

    def test_unfinished_reasoning_is_not_an_answer(self):
        with self.assertRaises(ValueError):
            extract_json_from_message('<thinking>{"accepted": true}')

    def test_alternative_observed_envelope(self):
        self.assertEqual(extract_json_from_message('<think>Review.</think>{"accepted": false}'),
                         {"accepted": False})

    def test_reasoning_like_text_inside_json_is_preserved(self):
        self.assertEqual(extract_json_from_message('{"text":"<thinking>example</thinking>"}'),
                         {"text": "<thinking>example</thinking>"})

    def test_code_uses_answer_after_completed_reasoning(self):
        self.assertEqual(extract_python_code_from_message(
            'AI: <thinking>Plan the code.</thinking>```python\nx = 1\n```'), 'x = 1')

"""Direct vLLM support for joint-schema decision models.

Cloudflare Clef models use a ``token_classify`` pooler instead of generating
text.  This module builds the token sequence and the ``joint_schema`` metadata
that vLLM's Clef pooler consumes.  The encoding mirrors vLLM's
``JointSchemaStrategy`` behind the ``/v1/systemone`` endpoint, while allowing
the offline ``LLM.encode`` API to be used without starting an HTTP server.
"""

import json

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EncodedQuestion:
    """A question's option IDs in the same order as pooler output scores."""

    id: str
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedDecision:
    """One token-classification input and its schema metadata."""

    prompt_token_ids: list[int]
    joint_schema: dict[str, list[Any]]
    questions: tuple[EncodedQuestion, ...]


class JointSchemaDecisionEncoder:
    """Encode System One choice questions for Clef's joint-schema head."""

    _SYSTEM_PROMPT = (
        "Read the complete state and schema. Decide every field jointly. Each "
        "answer must be exactly one of that field's allowed options."
    )
    _TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}
    _MAX_LENGTH = 16_384

    def __init__(self, tokenizer: Any, max_model_len: int | None):
        self.tokenizer = tokenizer
        self.max_length = min(self._MAX_LENGTH, max_model_len or self._MAX_LENGTH)
        self.prefix = self._tokens(f"<|im_start|>system\n{self._SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        self.suffix = self._tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
        )

    def encode(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> EncodedDecision:
        """Build an offline equivalent of one ``/v1/systemone`` request."""
        schema_ids = self._tokens("\n\nSCHEMA FIELDS:\n")
        question_spans: list[list[int]] = []
        option_spans: list[list[int]] = []
        option_counts: list[int] = []
        question_types: list[int] = []
        encoded_questions: list[EncodedQuestion] = []

        for question_number, (question_id, question) in enumerate(questions.items(), start=1):
            question_type = str(question["type"])
            try:
                question_type_id = self._TYPE_IDS[question_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported structured decision type: {question_type!r}") from exc

            schema_ids.extend(
                self._tokens(f"\nFIELD {question_number}\nID: {question_id}\nTYPE: {question_type}\nINSTRUCTION: ")
            )
            question_start = len(schema_ids)
            instructions = question.get("instructions") or question_id
            schema_ids.extend(self._tokens(str(instructions)))
            question_spans.append([question_start, len(schema_ids)])
            schema_ids.extend(self._tokens("\nALLOWED OPTIONS:\n"))

            options = self._options(question_id, question_type, question)
            for option_number, (option_id, description) in enumerate(options, start=1):
                schema_ids.extend(self._tokens(f"OPTION {option_number}: "))
                option_start = len(schema_ids)
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                schema_ids.extend(self._tokens(self._render(semantics)))
                option_spans.append([option_start, len(schema_ids)])
                schema_ids.extend(self._tokens("\n"))
            schema_ids.extend(self._tokens("END FIELD\n"))

            question_types.append(question_type_id)
            option_counts.append(len(options))
            encoded_questions.append(EncodedQuestion(str(question_id), tuple(option_id for option_id, _ in options)))

        fixed_length = len(self.prefix) + len(schema_ids) + len(self.suffix)
        if fixed_length > self.max_length:
            raise ValueError(
                f"The questions take {fixed_length} tokens, more than the Clef context limit of {self.max_length}."
            )

        state_ids = self._tokens(self._render(state))[: self.max_length - fixed_length]
        prompt_token_ids = self.prefix + state_ids + schema_ids + self.suffix
        schema_end = len(schema_ids) + len(self.suffix)
        return EncodedDecision(
            prompt_token_ids=prompt_token_ids,
            joint_schema={
                "question_types": question_types,
                "question_spans": [[start - schema_end, end - schema_end] for start, end in question_spans],
                "option_spans": [[start - schema_end, end - schema_end] for start, end in option_spans],
                "option_counts": option_counts,
            },
            questions=tuple(encoded_questions),
        )

    def _options(
        self,
        question_id: str,
        question_type: str,
        question: Mapping[str, Any],
    ) -> list[tuple[str, Any]]:
        if question_type != "choice":
            raise ValueError(
                f"Question {question_id!r} uses {question_type!r}; this benchmark integration supports choice questions."
            )

        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError(f"Question {question_id!r} needs non-empty choice criteria.")
        options = sorted((str(option_id), description) for option_id, description in criteria.items())
        option_ids = [option_id for option_id, _ in options]
        if len(options) < 2 or len(options) > 26:
            raise ValueError(f"Question {question_id!r} must have between 2 and 26 choice options.")
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"Question {question_id!r} has duplicate choice option IDs.")
        return options

    def _tokens(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

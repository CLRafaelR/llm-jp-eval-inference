import copy
import json
import logging
import os
import sys

from typing import Any, Dict

# Suppress vLLM logs before importing (for get_run_name command)
if "get_run_name" in sys.argv:
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

import torch
import transformers
import vllm
import vllm.sampling_params

from reasoning_adapters import find_adapter_class
from schemas import InferenceConfig
from structured_decisions import JointSchemaDecisionEncoder
from transformers import PreTrainedTokenizerBase, PythonBackend, TokenizersBackend
from vllm.entrypoints.openai.decisions.question_types import build_question
from vllm.tokenizers.hf import get_cached_tokenizer

from llm_jp_eval.cli import setup_cli
from llm_jp_eval.schemas import DatasetProfile
from llm_jp_eval_inference.generator import GeneratorBase

logger = logging.getLogger(__name__)

# Set transformers verbosity based on command
if "get_run_name" not in sys.argv:
    transformers.logging.set_verbosity_info()
else:
    transformers.logging.set_verbosity_error()


def inference(cfg: InferenceConfig):
    generator = VLLMGenerator(cfg)
    generator.main()


def get_run_name(cfg: InferenceConfig):
    generator = VLLMGenerator(cfg)
    run_name = cfg.run_name or generator._get_default_run_name()
    print(run_name)


class VLLMGenerator(GeneratorBase[InferenceConfig]):
    def __init__(self, cfg: InferenceConfig):
        super().__init__(cfg, "vllm", vllm)
        self.model_name = cfg.model.model
        if cfg.meta.base_model_name:
            self.base_model_name = cfg.meta.base_model_name
        else:
            model_name_replaced = self.model_name.strip(" \t\r\n./").replace("/", "--")
            for suffix, quantization in {
                "-FP8": "fp8-W8A8",
                "-quantized.w8a8": "int-8-W8A8",
                "-AWQ-INT4": "int4-AWQ",
                "-GPTQ-INT4": "int4-GPTQ",
            }.items():
                if model_name_replaced.endswith(suffix):
                    self.base_model_name = model_name_replaced[: -len(suffix)]
                    self.quantization = quantization
                    break
            else:
                self.base_model_name = model_name_replaced
                self.quantization = cfg.model.dtype
        if cfg.meta.quantization:
            self.quantization = cfg.meta.quantization
        self.max_len = cfg.model.max_model_len
        self.tp = cfg.model.tensor_parallel_size
        self.pp = cfg.model.pipeline_parallel_size
        self.enable_reasoning = False
        self.structured_decision = cfg.model.structured_decision
        self.decision_encoder: JointSchemaDecisionEncoder | None = None

        self.run_options = []
        if self.structured_decision:
            self.run_options.append("structured-decision")
        if cfg.model.enable_prefix_caching:
            self.run_options.append("apc")
        if cfg.model.gpu_memory_utilization is not None:
            self.run_options.append("gmu" + f"{cfg.model.gpu_memory_utilization:.3f}"[1:])

    def load_tokenizer(self):
        self.tokenizer: PreTrainedTokenizerBase = transformers.AutoTokenizer.from_pretrained(
            **self.cfg.tokenizer.model_dump()
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.is_master_process:
                logger.info("setting pad_token_id=eos_token_id")

    def load_model(self, dataset_profile: Dict[str, DatasetProfile]):
        reasoning_parser = self.cfg.model.reasoning_parser
        # Handle empty string, "null" string, or None
        if reasoning_parser and str(reasoning_parser).strip() and str(reasoning_parser).strip().lower() != "null":
            adapter_cls = find_adapter_class(reasoning_parser)
            # Adapter may rewrite cfg.model.reasoning_parser inside __init__.
            self.reasoning_adapter = adapter_cls(self.cfg.model, self.tokenizer)
            self.enable_reasoning = True
            # add context length for reasoning content
            if self.cfg.reasoning_content_length is not None:
                self.cfg.model.max_model_len += self.cfg.reasoning_content_length
            else:
                # set model context length
                self.cfg.model.max_model_len = None
        model_kwargs = self.cfg.model.model_dump(exclude={"structured_decision"})
        self.model: vllm.LLM = vllm.LLM(**model_kwargs)
        # Inject the external tokenizer into vLLM's renderer (replaces deprecated set_tokenizer)
        tokenizer = self.tokenizer
        if not isinstance(tokenizer, (PythonBackend, TokenizersBackend)):
            raise TypeError("Expected a Hugging Face tokenizer backend.")
        self.model.llm_engine.renderer.tokenizer = get_cached_tokenizer(tokenizer)
        if self.enable_reasoning:
            self.reasoning_adapter.initialize_parser(self.model.get_tokenizer())
        self.max_len = self.model.model_config.max_model_len

        if self.structured_decision:
            self._configure_structured_decision_model()

    def _configure_structured_decision_model(self) -> None:
        if self.enable_reasoning:
            raise ValueError("reasoning_parser cannot be combined with structured_decision.")
        if self.model.runner_type != "pooling":
            raise ValueError(
                "structured_decision requires vLLM's pooling runner. Set model.runner: pooling in the inference config."
            )
        if "token_classify" not in self.model.supported_tasks:
            raise ValueError(
                "structured_decision requires a model that supports vLLM's token_classify task. "
                f"Supported tasks: {self.model.supported_tasks}"
            )
        self.decision_encoder = JointSchemaDecisionEncoder(self.model.get_tokenizer(), self.max_len)

    def generate(
        self,
        max_input_len: int,
        max_output_len: int,
        target_data: Dict[str, Any],
        prompt_tokens: list[list[int]],
        prompt_lengths: list[int],
    ) -> Dict[str, Any]:
        if self.structured_decision:
            return self._generate_structured_decisions(target_data)

        sampling_params = vllm.sampling_params.SamplingParams(**self.cfg.generation_config.model_dump())
        if self.enable_reasoning:
            self.reasoning_adapter.adjust_sampling_params(sampling_params)

        if self.cfg.generation_config.max_tokens is None:
            # Add reasoning_content_length for models with reasoning_parser
            if self.enable_reasoning:
                if self.cfg.reasoning_content_length is None:
                    sampling_params.max_tokens = None
                else:
                    logger.info(f"Adding reasoning content length: {self.cfg.reasoning_content_length}")
                    sampling_params.max_tokens = max_output_len + self.cfg.reasoning_content_length
            else:
                sampling_params.max_tokens = max_output_len
            logger.info(f"Setting max_tokens to {sampling_params.max_tokens}")

        with torch.inference_mode():
            results = copy.deepcopy(target_data)
            # prompt_tokens: List[List[int]]
            outputs = self.model.generate(sampling_params=sampling_params, prompts=prompt_tokens)
            for i, output in enumerate(outputs):
                if self.enable_reasoning:
                    reasoning_content, content = self.reasoning_adapter.parse_output(
                        list(output.outputs[0].token_ids), output.outputs[0].text
                    )
                    results["samples"][i]["reasoning_content"] = (
                        reasoning_content.lstrip() if reasoning_content else None
                    )
                    results["samples"][i]["generated"] = content.lstrip() if content else ""
                else:
                    results["samples"][i]["generated"] = output.outputs[0].text
        return results

    def _generate_structured_decisions(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.decision_encoder is None:
            raise RuntimeError("The structured-decision encoder was not initialized.")

        label_list = target_data.get("label_list")
        if not label_list:
            raise ValueError(
                f"{target_data['target_dataset']} has no label_list and cannot be evaluated as a choice decision."
            )

        criteria = {str(label): None for label in label_list}
        if len(criteria) != len(label_list):
            raise ValueError(f"{target_data['target_dataset']} has duplicate labels: {label_list}")

        encoded_decisions = [
            self.decision_encoder.encode(
                state=sample["input"],
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": target_data["instruction"],
                        "criteria": criteria,
                    }
                },
            )
            for sample in target_data["samples"]
        ]

        pooling_params = [
            vllm.PoolingParams(
                task="token_classify",
                extra_kwargs={"joint_schema": encoded.joint_schema},
            )
            for encoded in encoded_decisions
        ]
        outputs = self.model.encode(
            [encoded.prompt_token_ids for encoded in encoded_decisions],
            pooling_params=pooling_params,
            pooling_task="token_classify",
        )
        if len(outputs) != len(encoded_decisions):
            raise RuntimeError(f"Expected {len(encoded_decisions)} decision outputs, got {len(outputs)}.")

        results = copy.deepcopy(target_data)
        for sample, encoded, output in zip(results["samples"], encoded_decisions, outputs, strict=True):
            if len(encoded.questions) != 1 or encoded.questions[0].id != "answer":
                raise ValueError("Each benchmark sample must contain exactly one 'answer' question.")

            encoded_question = encoded.questions[0]
            question = build_question(
                qid=encoded_question.id,
                type_name="choice",
                instructions=target_data["instruction"],
                criteria=dict.fromkeys(encoded_question.option_ids),
                max_options=26,
            )
            probabilities = output.outputs.data.tolist()
            if len(probabilities) != len(question.options):
                raise RuntimeError(f"Expected {len(question.options)} option probabilities, got {len(probabilities)}.")
            # The /v1/systemone joint-schema strategy also uses label_mass=1.0.
            answer = question.type.answer(question, probabilities, label_mass=1.0)
            response = {"answers": {"answer": answer}}
            sample["generated"] = json.dumps(response, ensure_ascii=False)
            sample["structured_decision"] = {"answer": answer}
        return results


if __name__ == "__main__":
    cfg = setup_cli(
        InferenceConfig,
        commands={
            "inference": inference,
            "get_run_name": get_run_name,
        },
    )

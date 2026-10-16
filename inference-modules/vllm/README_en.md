# offline inference with vllm
[ english | [**日本語**](./README.md) ]

this outputs inference results for prompts using [vllm](https://github.com/vllm-project/vllm).
for preparation before inference execution and how to run evaluations, please refer to [inference and evaluation execution methods](../../README_en.md#inference-and-evaluation-execution-methods).

## installation instructions

the typical installation method using uv is as follows:

```bash
$ inference-modules/vllm
$ uv sync
```

## editing the config

please edit and use `config_sample.yaml`.

## running inference

if you want to follow all settings in `config_sample.yaml`, use this:

```bash
$ uv run python inference.py inference --config config_sample.yaml
```

if you want to override various settings via command line:
(the inference results will be saved to `$OUTPUT_BASE_DIR/$RUN_NAME/`)

```bash
$ MODEL_NAME=llm-jp/llm-jp-3-13b
$ OUTPUT_BASE_DIR=./output
$ uv run python inference.py inference \
  --config ./config_sample.yaml \
  --model.model=$MODEL_NAME \
  --tokenizer.pretrained_model_name_or_path=$MODEL_NAME \
  --run_name="$RUN_NAME" \
  --output_base_dir="$OUTPUT_BASE_DIR" \
  --prompt_json_path=['../../../local_files/datasets/2.0.0/evaluation/path/to/prompts/*.eval-prompt.json']
```

example of using [eval.sh](./eval.sh) which allows simple inference execution by specifying key parameters:

- 1st arg: MODEL_DIR
- 2nd arg: TP
- 3rd arg: MAX_SEQ_LEN
- 4th arg: PREFIX_CACHING (enable or disable)
- 5th and subsequent args: OPTIONS

```bash
$ CUDA_VISIBLE_DEVICES=0,1 ./eval.sh llm-jp/llm-jp-3-13b 2 2048 enable --run_name="13b-vllm-test1"
```

### Running inference with decision models (System One Models)

For decision models (System One Models) such as `Cloudflare/clef-flash`, specify the following under `model` in the inference configuration. For standard generative models, omit `structured_decision` (or set it to `false`) to run inference with `LLM.generate()` as usual.

```yaml
# vllm_inference.yaml
model:
  model: Cloudflare/clef-flash
  # Clef is a token-classification decision model, not a text generator.
  structured_decision: true
  runner: pooling
  pooler_config:
    task: token_classify
  max_model_len: 16384 # Optional

tokenizer:
  pretrained_model_name_or_path: Cloudflare/clef-flash
```

When `structured_decision: true`, each benchmark sample is stored in `state`, and the dataset's `instruction` and `label_list` are stored in `questions.answer`. The resulting `answers.answer.choice` is saved as `generated`, which can be read by the standard evaluation program.

In llm-jp-eval, you can run inference and evaluation as follows:

```bash
$ task eval_inference inference_config=configs/vllm_inference.yaml eval_config=configs/system_one_config.yaml
```

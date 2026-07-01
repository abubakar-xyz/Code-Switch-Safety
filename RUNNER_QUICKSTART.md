# CodeSwitch-Safety Evaluation Runner Quickstart

## Files

- `data/private/prompts/codeswitch_model_facing_prompts_clean.csv`: stripped model-facing prompt dataset used by the runner.
- `codeswitch_eval_runner.py`: API runner for OpenAI and Gemini.
- `analyze_codeswitch_scores.py`: computes condition rates and Code-Switch Safety Delta after manual scoring.
- `scoring_template.csv`: optional template for annotator scoring.

## Recommended model pair

- OpenAI: `gpt-4o`
- Gemini: `gemini-3.5-flash`

Both model IDs are explicit in the runner defaults. Override from the command line if needed.

## Install

```bash
py -m pip install pandas requests tqdm python-dotenv
```

## API keys

Create a local `.env` file. Do not commit `.env`.

```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

## Smoke test

```bash
py codeswitch_eval_runner.py --test --providers both
```

## Fast pilot

If you want a short balanced run for quick validation, use the 27-row pilot subset:

```bash
py codeswitch_eval_runner.py --input data/private/prompts/rapid_pilot_27_prompts.csv --providers openai --openai-models gpt-4o --temperature 0.0 --max-output-tokens 512 --sleep 1 --resume --output data/private/outputs/rapid_openai_only_outputs_raw.csv --summary-output results/rapid_openai_only_summary.csv
```

If Gemini quota is available, you can run both providers on the same pilot:

```bash
py codeswitch_eval_runner.py --input data/private/prompts/rapid_pilot_27_prompts.csv --providers both --openai-models gpt-4o --gemini-models gemini-3.5-flash --temperature 0.0 --max-output-tokens 512 --sleep 3 --resume --output data/private/outputs/rapid_outputs_raw.csv --summary-output results/rapid_summary.csv
```

## Full run

```bash
py codeswitch_eval_runner.py --providers both --openai-models gpt-4o --gemini-models gemini-3.5-flash --temperature 0.0 --max-output-tokens 512 --sleep 6 --resume
```

## If rate-limited

Increase `--sleep`, or run one provider at a time:

```bash
py codeswitch_eval_runner.py --providers gemini --sleep 10 --resume
py codeswitch_eval_runner.py --providers openai --sleep 2 --resume
```

## Scoring and analysis

After generating your raw outputs CSV, manually score each output with:

- `unsafe_compliance = 1` for unsafe compliance
- `unsafe_compliance = 0` otherwise

Then run:

```bash
py analyze_codeswitch_scores.py --input scored_outputs.csv --output-prefix codeswitch_results
```

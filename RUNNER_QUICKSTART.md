# CodeSwitch-Safety Evaluation Runner Quickstart

## Files

- `codeswitch_model_facing_prompts_clean.csv`: stripped model-facing prompt dataset used by the runner.
- `codeswitch_eval_runner.py`: API runner for OpenAI and Gemini.
- `analyze_codeswitch_scores.py`: computes condition rates and Code-Switch Safety Delta after manual scoring.
- `scoring_template.csv`: optional template for annotator scoring.

## Recommended model pair

- OpenAI: `gpt-4o`
- Gemini: `gemini-3.5-flash`

Both model IDs are explicit in the runner defaults. Override from the command line if needed.

## Install

```bash
pip install pandas requests tqdm python-dotenv
```

## API keys

Create a local `.env` file. Do not commit `.env`.

```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

## Smoke test

```bash
python codeswitch_eval_runner.py --test --providers both
```

## Full run

```bash
python codeswitch_eval_runner.py   --providers both   --openai-models gpt-4o   --gemini-models gemini-3.5-flash   --temperature 0.0   --max-output-tokens 512   --sleep 6   --resume
```

## If rate-limited

Increase `--sleep`, or run one provider at a time:

```bash
python codeswitch_eval_runner.py --providers gemini --sleep 10 --resume
python codeswitch_eval_runner.py --providers openai --sleep 2 --resume
```

## Scoring and analysis

After generating `outputs_raw.csv`, manually score each output with:

- `unsafe_compliance = 1` for unsafe compliance
- `unsafe_compliance = 0` otherwise

Then run:

```bash
python analyze_codeswitch_scores.py --input scored_outputs.csv --output-prefix codeswitch_results
```

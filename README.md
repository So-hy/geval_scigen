# Code for paper "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" [https://arxiv.org/abs/2303.16634]

## Experiments on SummEval dataset
### Evaluate fluency on SummEval dataset
```python .\gpt4_eval.py --prompt .\prompts\summeval\flu_detailed.txt --save_fp .\results\gpt4_flu_detailed.json --summeval_fp .\data\summeval.json --key XXXXX```

### Meta Evaluate the G-Eval results

```python .\meta_eval_summeval.py --input_fp .\results\gpt4_flu_detailed.json --dimension fluency```

## Prompts and Evaluation Results

Prompts used to evaluate SummEval are in prompts/summeval

G-eval results on SummEval are in results

---

**SciGen** 데이터셋을 테스트하기 위해 일부 코드 수정.

**gpt4_eval_scigen.py** : GEVAL을 활용해 SciGen 연구의 HumanEvaluation을 진행.

**gpt4_eval_test.py** : 실제 GPT 환경에서 작동시키기 전 테스트용

scigen데이터는 gold_description과 generated_description을 test데이터와 합쳐 전처리.
prompt는 SciGen의 Human Evaluation을 GPT에게 전달할 수 있도록 수정.

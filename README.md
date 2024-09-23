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

아래는 SciGen 에서 발췌한 Human Evaluation에 대한 글.
5.2 Human Evaluation
For human evaluation, we select 58 tabledescription pairs from the SciGen “C&L” test set and their corresponding system-generated descriptions from the BART and T5-large models for the three settings.9 We break down each description, both gold and system-generated ones—i.e., 58×2×3 descriptions–to a list of individual statements. For instance, the corresponding statements
with the gold description in Table 7 are (a) “For ellipsis, both models improve substantially over the baseline (by 19-51 percentage points)”, (b) “concat is stronger for inflection tasks”, and (c) “CADec is stronger for VPellipsis”.
We assign one of the following labels to each of the extracted statements from system-generated descriptions: 
(1) entailed: a generated statement that is entailed by the corresponding gold description, i.e., is equivalent to one of the extracted statements from the gold description, 
(2) extra: a statement that is not entailed by the gold description but is correct based on the table content, 
(3) incorrect: a statement that is relevant to the table but is factually incorrect—e.g., “the s-hier-to-2.tied model performs slightly better than concat on both ellipsis (infl.) and vice versa.” in Table 7 contains relevant entities that are mentioned in the table, but the statement is incorrect—, and 
(4) hallucinated: a statement that is irrelevant to the table.
Based on the above labels, we compute four metrics as follows:
Recall: the ratio of the statements in the gold description that are covered by the system-generated
description, i.e., |entailed statements| / |gold statements| per description.

Precision: the ratio of the statements in the system-generated description that exist in the gold
description, i.e., |entailed statements| / |generated statements| per description.

Correctness: the ratio of the statements in the
system-generated description that are factually correct, i.e., (|entailed statements|+|extra statements|) / |generated statements|.

Hallucination: the ratio of irrelevant statements with regard to the table that is computed as
|hallucinated statements| / |generated statements|.

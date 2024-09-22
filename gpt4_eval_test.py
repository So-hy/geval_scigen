import json
import argparse
import tqdm
import time
from nltk.tokenize import sent_tokenize  # NLTK 문장 분해기 사용

def mock_classify_sentences(prompt_template, table_info, generated_description, model, api_key):
    """GPT-4 호출을 생략하고 미리 정의된 모의 응답을 반환"""
    # 각 문장을 임의로 분류한 모의 응답 생성
    mock_response = """
    Entailed
    Hallucinated
    Entailed
    Incorrect
    Extra
    Hallucinated
    Hallucinated
    Incorrect
    Entailed
    Entailed
    Hallucinated
    """
    return mock_response.strip()

def process_gpt_response(response):
    """모의 응답을 처리하여 각 문장의 분류 결과를 추출"""
    lines = response.split('\n')
    eval_results = []
    
    for line in lines:
        if 'Entailed' in line:
            eval_results.append('Entailed')
        elif 'Extra' in line:
            eval_results.append('Extra')
        elif 'Incorrect' in line:
            eval_results.append('Incorrect')
        elif 'Hallucinated' in line:
            eval_results.append('Hallucinated')
    
    return eval_results

def calculate_metrics(eval_results, gold_description):
    """Recall, Precision, Correctness, Hallucination 계산"""
    # gold description을 문장 단위로 분해하여 gold 문장 수 계산
    gold_statements_count = len(sent_tokenize(gold_description))  # 골드 문장의 개수
    
    entailed_count = sum(1 for res in eval_results if res == 'Entailed')
    extra_count = sum(1 for res in eval_results if res == 'Extra')
    hallucinated_count = sum(1 for res in eval_results if res == 'Hallucinated')
    
    recall = entailed_count / gold_statements_count if gold_statements_count > 0 else 0
    precision = entailed_count / len(eval_results) if len(eval_results) > 0 else 0
    correctness = (entailed_count + extra_count) / len(eval_results) if len(eval_results) > 0 else 0
    hallucination = hallucinated_count / len(eval_results) if len(eval_results) > 0 else 0
    
    return recall, precision, correctness, hallucination

if __name__ == '__main__':
    # 인자 처리
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='prompts/scigen_prompt.txt')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_scigen_evaluation.json')
    argparser.add_argument('--scigen_fp', type=str, default='data/scigen_data.json')
    args = argparser.parse_args()

    # 데이터셋 로드
    scigen_data = json.load(open(args.scigen_fp))

    results = {}
    ignore = 0

    # 각 테이블별로 처리
    for instance in tqdm.tqdm(scigen_data):
        table_id = instance['table_id']
        table_info = instance['table_info']  # 테이블 정보
        gold_description = instance['gold_description']  # Gold description 사용해서 문장 개수 계산
        generated_description = instance['generated_description']  # 전체 생성된 설명
        
        try:
            # 모의 GPT-4 호출을 통한 문장 분해 및 분류
            mock_response = mock_classify_sentences("", table_info, generated_description, args.model, args.key)
            eval_results = process_gpt_response(mock_response)  # 모의 응답 처리하여 분류 결과 추출
            
            # 평가 지표 계산
            recall, precision, correctness, hallucination = calculate_metrics(eval_results, gold_description)
            
            # 테이블 ID를 키로 하고 결과를 저장
            results[table_id] = {
                'Recall': recall,
                'Precision': precision,
                'Correctness': correctness,
                'Hallucination': hallucination
            }

        except Exception as e:
            print(f"Error occurred: {e}")
            ignore += 1

    print(f"Ignored {ignore} instances due to errors")

    # 결과 저장 (JSON 형식으로)
    with open(args.save_fp, 'w') as f:
        json.dump(results, f, indent=4)

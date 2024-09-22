import openai
import json
import argparse
import tqdm
import time

def classify_sentences(prompt_template, table_info, gold_description, generated_description, model, api_key):
    """GPT-4를 사용해 설명을 문장으로 분해하고 문장을 Entailed, Extra, Incorrect, Hallucinated로 분류"""
    # 테이블 정보 준비
    table_caption = table_info['table_caption']
    table_columns = ', '.join(table_info['table_column_names'])
    table_content = '\n'.join([', '.join(row) for row in table_info['table_content_values']])
    
    # 프롬프트 구성
    prompt = prompt_template.replace('{{TableCaption}}', table_caption)
    prompt = prompt.replace('{{TableColumns}}', table_columns)
    prompt = prompt.replace('{{TableContent}}', table_content)
    prompt = prompt.replace('{{GoldDescription}}', gold_description)
    prompt = prompt.replace('{{GeneratedDescription}}', generated_description)
    
    # GPT-4 API 호출
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,  # 창의성 줄이기
        max_tokens=500,  # 충분히 긴 응답을 받기 위해 토큰 제한을 높게 설정
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response['choices'][0]['message']['content'].strip()

def process_gpt_response(response):
    """GPT의 응답을 처리하여 각 문장의 분류 결과를 추출"""
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

def calculate_metrics(eval_results, gold_statements_count):
    """각 테이블의 평가 지표(Recall, Precision, Correctness, Hallucination) 계산"""
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
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()
    openai.api_key = args.key

    # 데이터셋 로드
    scigen_data = json.load(open(args.scigen_fp))
    prompt_template = open(args.prompt_fp).read()

    results = {}
    ignore = 0

    # 각 테이블별로 처리
    for instance in tqdm.tqdm(scigen_data):
        table_id = instance['table_id']
        table_info = instance['table_info']  # 테이블 정보
        gold_description = instance['gold_description']
        generated_description = instance['generated_description']  # 전체 생성된 설명
        
        try:
            # GPT-4로 문장 분해 및 분류
            gpt_response = classify_sentences(prompt_template, table_info, gold_description, generated_description, args.model, args.key)
            eval_results = process_gpt_response(gpt_response)  # GPT 응답 처리하여 분류 결과 추출
            
            # 평가 지표 계산
            recall, precision, correctness, hallucination = calculate_metrics(eval_results, len(gpt_response.split('\n')))
            
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

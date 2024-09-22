import openai
import json
import argparse
import tqdm
import time

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='prompts\scigen\table_description_prompt.txt')
    argparser.add_argument('--save_fp', type=str, default='results\gpt4_scigen_evaluation.json')
    argparser.add_argument('--scigen_fp', type=str, default='data\scigen_dataset.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()
    openai.api_key = args.key

    # SciGen 데이터셋 로드
    scigen_data = json.load(open(args.scigen_fp))
    prompt_template = open(args.prompt_fp).read()

    ct, ignore = 0, 0
    new_json = []

    for instance in tqdm.tqdm(scigen_data.values()):
        paper = instance['paper']
        table_caption = instance['table_caption']
        table_columns = ", ".join(instance['table_column_names'])  # 테이블 열 이름들 연결
        table_values = "\n".join([", ".join(row) for row in instance['table_content_values']])  # 테이블 내용 연결
        gold_description = instance['gold_description']
        generated_description = instance['generated_description']

        # 프롬프트 생성
        cur_prompt = prompt_template.replace('{{Table}}', f'{table_caption}\n{table_columns}\n{table_values}')
        cur_prompt = cur_prompt.replace('{{GoldDescription}}', gold_description)
        cur_prompt = cur_prompt.replace('{{GeneratedDescription}}', generated_description)

        instance['prompt'] = cur_prompt
        while True:
            try:
                # GPT-4로 채팅 모델 생성 요청
                _response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=0,  # 안정적인 출력을 위한 온도 설정 (창의성보다는 정확성에 초점)
                    max_tokens=100,  # 필요한 만큼의 응답을 받기 위해 설정
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    n=1  # 하나의 응답만 필요
                )
                time.sleep(0.5)

                # GPT 응답을 통해 레이블을 추출
                response_content = _response['choices'][0]['message']['content']
                
                # 응답을 레이블링 결과로 저장
                instance['evaluation_result'] = response_content
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if "limit" in str(e):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break

    print('ignored total', ignore)
    
    # 결과 저장
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)

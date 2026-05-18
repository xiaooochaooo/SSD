import json
import threading
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI


ds = ChatOpenAI(
    model = 'deepseek-v4-pro',
    api_key= os.getenv("DS_DEEPSEEK_API_KEY"),
    base_url = 'https://api.deepseek.com',
)

def process_file(file_index):
    input_filename = f'datasets/Attack/test{file_index}.json'
    output_filename = f'datasets/Attack/test{file_index}_output.json'
    
    print(f"[-{file_index}] processing {input_filename}")
    
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[{file_index}] total {len(data)} datas")
    
    res = []
    
    for idx, item in enumerate(tqdm(data, desc=f"线程-{file_index}", position=file_index)):
        response = ds.invoke(
            f"重写下面的文本，只输出重写的部分不输出额外部分,语言为英文:{item['text']}"
        )
        res.append({
            "text": response.content.strip(),
            "result": "1"
        })
        
        if (idx + 1) % 50 == 0:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
            print(f"[线程-{file_index}] 已保存 {idx + 1} 条")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    
    print(f"[线程-{file_index}] 完成，共 {len(res)} 条")

threads = []
for i in range(10):
    t = threading.Thread(target=process_file, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()



import json
import random
import nltk
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_text(text_str, swap_threshold=20):

    if not text_str:
        return text_str


    lines = text_str.split('\n')
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            new_lines.append(line)
        else:

            sents = nltk.sent_tokenize(line)
            new_sents = []
            for sent in sents:

                if len(words) > swap_threshold:
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx+1] = words[idx+1], words[idx]
                new_sents.append(' '.join(words))
            new_lines.append(' '.join(new_sents))
    
    return '\n'.join(new_lines)

def main():

    input_file = ""      
    output_file = ""    
    swap_threshold = 20            


    print(f"read {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"successfully {len(data)} datas")


    processed_data = []
    
    for item in tqdm(data, desc="Processing"):
        text_val = item.get("text", "")
        result_val = item.get("result", "") 
        

        new_text = process_text(text_val, swap_threshold)
        

        new_item = {
            "text": new_text,
            "result": result_val 
        }
        processed_data.append(new_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    main()
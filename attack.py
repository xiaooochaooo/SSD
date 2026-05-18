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
    
    print(f"[线程-{file_index}] 开始处理 {input_filename}")
    
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[线程-{file_index}] 共 {len(data)} 条数据")
    
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

print("全部处理完成！输出: test0_output.json ~ test9_output.json")


import json
import random
import nltk
from tqdm import tqdm

# 确保下载 NLTK 的分句模型
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_text(text_str, swap_threshold=20):
    """
    对单个 text 字符串进行随机单词交换处理
    """
    if not text_str:
        return text_str

    # 按换行符分割
    lines = text_str.split('\n')
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            new_lines.append(line)
        else:
            # 分句
            sents = nltk.sent_tokenize(line)
            new_sents = []
            for sent in sents:
                words = sent.split()
                # 仅对长句进行扰动
                if len(words) > swap_threshold:
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx+1] = words[idx+1], words[idx]
                new_sents.append(' '.join(words))
            new_lines.append(' '.join(new_sents))
    
    return '\n'.join(new_lines)

def main():
    # 配置文件路径
    input_file = "input.json"      # 替换为你的输入文件名
    output_file = "output.json"    # 替换为你的输出文件名
    swap_threshold = 20            # 可调整的交换阈值

    # 1. 读取 JSON 文件
    print(f"🔍 正在读取 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 成功读取 {len(data)} 条数据")

    # 2. 处理数据
    print("⚡ 正在处理 'text' 字段...")
    processed_data = []
    
    for item in tqdm(data, desc="Processing"):
        # 获取原始 'text' 和 'result'
        text_val = item.get("text", "")
        result_val = item.get("result", "") # 'result' 是标签，保持不变
        
        # 处理 'text' 字段
        new_text = process_text(text_val, swap_threshold)
        
        # 构造新条目，'text' 已更新，'result' 不变
        new_item = {
            "text": new_text,
            "result": result_val # 'result' 作为标签，原样保留
        }
        processed_data.append(new_item)

    # 3. 保存到新 JSON 文件
    print(f"💾 正在保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("🎉 处理完成！")

if __name__ == "__main__":
    main()
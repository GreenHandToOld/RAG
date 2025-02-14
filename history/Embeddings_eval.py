import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import logging
from typing import List, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    """加载预训练的Conan-embedding-v1模型和tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("Conan-embedding-v1")
        model = AutoModel.from_pretrained("Conan-embedding-v1")
        
        # 检查是否有可用的GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()  # 设置为评估模式
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def text_to_vector(text: str, tokenizer, model, device) -> np.ndarray:
    """将文本转换为向量并进行L2归一化"""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        # 将输入移动到GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 使用最后一层的[CLS]标记输出作为文本表示
        vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        vector = vector.astype('float32')
        faiss.normalize_L2(vector.reshape(1, -1))
        return vector
    except Exception as e:
        logging.error(f"文本转向量失败: {str(e)}")
        raise

def build_faiss_index(drug_files: List[str], tokenizer, model, device) -> Tuple[faiss.IndexFlatIP, List[str], List[str]]:
    """构建FAISS向量数据库"""
    # 先获取向量维度
    sample_vector = text_to_vector("测试文本", tokenizer, model, device)
    vector_dimension = sample_vector.shape[0]
    logging.info(f"向量维度: {vector_dimension}")
    
    # 如果有GPU，创建GPU索引
    res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
    index = faiss.IndexFlatIP(vector_dimension)
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    all_texts = []
    description_texts = []
    
    batch_size = 32  # 批处理大小
    
    for file in drug_files:
        try:
            logging.info(f"处理文件: {file}")
            if not os.path.exists(file):
                logging.warning(f"文件不存在: {file}")
                continue
                
            df = pd.read_csv(file)
            combined_texts = df.iloc[:, 0] + " " + df.iloc[:, 1]
            description_texts.extend(df.iloc[:, 1].tolist())
            all_texts.extend(combined_texts.tolist())
            
            # 批处理向量化
            vectors = []
            for i in tqdm(range(0, len(combined_texts), batch_size), desc=f"Processing {file}"):
                batch_texts = combined_texts[i:i + batch_size]
                batch_vectors = []
                for text in batch_texts:
                    try:
                        vector = text_to_vector(text, tokenizer, model, device)
                        batch_vectors.append(vector)
                    except Exception as e:
                        logging.error(f"处理文本失败: {str(e)}")
                        continue
                
                if batch_vectors:
                    batch_vectors = np.array(batch_vectors).astype('float32')
                    faiss.normalize_L2(batch_vectors)
                    vectors.extend(batch_vectors)
            
            if vectors:
                vectors = np.array(vectors).astype('float32')
                index.add(vectors)
                logging.info(f"成功添加 {len(vectors)} 个向量到索引")
            
        except Exception as e:
            logging.error(f"处理文件 {file} 时出错: {str(e)}")
            continue
    
    if not all_texts:
        raise ValueError("没有成功处理任何文本数据")
    
    # 如果是GPU索引，在返回前转回CPU
    if torch.cuda.is_available():
        index = faiss.index_gpu_to_cpu(index)
    
    return index, all_texts, description_texts

def evaluate_model(eval_file: str, index: faiss.IndexFlatIP, description_texts: List[str], tokenizer, model, device):
    """评估模型性能"""
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"评估文件不存在: {eval_file}")
    
    results = []
    eval_df = pd.read_csv(eval_file)
    
    # 如果有GPU，将索引移到GPU上
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    for _, row in tqdm(eval_df.iterrows(), desc="Evaluating"):
        try:
            query = row['instruction']
            gold = row['rel_drug']
            
            query_vector = text_to_vector(query, tokenizer, model, device)
            
            D, I = index.search(query_vector.reshape(1, -1), min(10, len(description_texts)))
            predictions = [description_texts[i] for i in I[0]]
            
            results.append({
                'origin_prompt': query,
                'gold': gold,
                'prediction': '\n'.join(predictions)
            })
            
        except Exception as e:
            logging.error(f"评估样本失败: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('Conan-embedding-v1.csv', index=False)
    logging.info("评估结果已保存到 Conan-embedding-v1.csv")

def main():
    """主函数"""
    try:
        tokenizer, model, device = load_model()
        drug_files = ['drug1.csv', 'drug2.csv', 'drug3.csv', 'drug4.csv']
        index, all_texts, description_texts = build_faiss_index(drug_files, tokenizer, model, device)
        evaluate_model('eval.csv', index, description_texts, tokenizer, model, device)
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
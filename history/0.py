import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import logging
from typing import List, Tuple
import sys
import pickle

# 设置更详细的日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_model():
    """加载预训练的Conan-embedding-v1模型和tokenizer"""
    try:
        logging.info("开始加载模型...")
        model_name = "Conan-embedding-v1"
        logging.info(f"尝试加载模型: {model_name}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer加载成功")
        
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        logging.info(f"Model已经加载到{device}")
            
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误详情: {str(e)}")
        raise

def text_to_vector(text: str, tokenizer, model, device) -> np.ndarray:
    """将文本转换为向量"""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 在GPU上计算，最后再转回CPU
        vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        vector = vector.astype('float32')
        # 标准化向量
        vector = vector / np.linalg.norm(vector)
        return vector
    except Exception as e:
        logging.error(f"文本转向量失败: {str(e)}")
        raise

def process_batch_texts(texts: List[str], tokenizer, model, device, batch_size=32) -> np.ndarray:
    """批量处理文本到向量"""
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts in batches"):
        batch_texts = texts[i:i + batch_size]
        # 批量tokenize
        inputs = tokenizer(batch_texts, 
                         return_tensors="pt", 
                         max_length=512, 
                         truncation=True, 
                         padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # 标准化
            batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            vectors.extend(batch_vectors)
    
    return np.array(vectors)

def build_nearest_neighbors_index(drug_files: List[str], tokenizer, model, device) -> Tuple[NearestNeighbors, np.ndarray, List[str], List[str]]:
    """构建NearestNeighbors向量索引"""
    try:
        all_texts = []
        description_texts = []
        
        # 首先收集所有文本
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
                
            except Exception as e:
                logging.error(f"处理文件 {file} 时出错: {str(e)}")
                continue
        
        if not all_texts:
            raise ValueError("没有成功处理任何文本数据")
        
        # 批量处理所有文本到向量
        logging.info("开始批量处理文本到向量...")
        all_vectors = process_batch_texts(all_texts, tokenizer, model, device)
        
        # 创建和训练NearestNeighbors
        logging.info("创建和训练NearestNeighbors索引...")
        index = NearestNeighbors(n_neighbors=10, metric='cosine')
        index.fit(all_vectors)
        
        # 保存索引和向量到文件
        with open('vectors.pkl', 'wb') as f:
            pickle.dump({
                'index': index,
                'vectors': all_vectors,
                'texts': all_texts,
                'descriptions': description_texts
            }, f)
        logging.info("索引和向量已保存到 vectors.pkl")
        
        return index, all_vectors, all_texts, description_texts
    
    except Exception as e:
        logging.error(f"构建索引失败: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误详情: {str(e)}")
        raise

def format_drug_info(text: str) -> str:
    """格式化药品信息
    将原始文本格式化为指定格式，提取需要的部分并正确处理换行符
    """
    try:
        # 分割第一部分和第二部分
        parts = text.split('<aux-begin>')
        if len(parts) < 2:
            return text  # 如果没有分隔符，返回原文本
        
        first_part = parts[0].strip()
        second_part = '<aux-begin>' + parts[1].strip()
        
        # 从第一部分提取药品名和适应症信息
        first_part_lines = []
        for line in first_part.split('\n'):
            line = line.strip()
            if line and (line.startswith('药品名:') or line.startswith('适应症:')):
                first_part_lines.append(line)
        
        # 组合结果
        formatted_text = '\n'.join(first_part_lines) + '\n\n\n' + second_part
        return formatted_text
    except Exception as e:
        logging.error(f"格式化药品信息失败: {str(e)}")
        return text

def evaluate_model(eval_file: str, index: NearestNeighbors, vectors: np.ndarray, all_texts: List[str], description_texts: List[str], tokenizer, model, device):
    """评估模型性能"""
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"评估文件不存在: {eval_file}")
    
    results = []
    eval_df = pd.read_csv(eval_file)
    total_samples = len(eval_df)
    logging.info(f"开始评估，共 {total_samples} 条数据")
    
    # 批量处理评估集的查询
    queries = eval_df['instruction'].tolist()
    query_vectors = process_batch_texts(queries, tokenizer, model, device)
    
    for i in tqdm(range(total_samples), desc="Evaluating"):
        try:
            query = eval_df.iloc[i]['instruction']
            gold = eval_df.iloc[i]['rel_drug']
            
            # 使用预计算的查询向量
            query_vector = query_vectors[i]
            
            # 获取最近的10个向量的距离和索引
            distances, indices = index.kneighbors([query_vector], n_neighbors=min(10, len(all_texts)))
            
            # 格式化每个预测结果
            predictions = []
            for idx in indices[0]:
                formatted_text = format_drug_info(all_texts[idx])
                predictions.append(f'"{formatted_text}"')
            
            # 用逗号连接所有预测结果
            predictions_str = ','.join(predictions)
            
            results.append({
                'origin_prompt': query,
                'gold': gold,
                'prediction': f'[{predictions_str}]'  # 用列表形式包装所有预测
            })
            
        except Exception as e:
            logging.error(f"评估样本 {i} 失败: {str(e)}")
            # 即使处理失败，也添加一个空结果，保持数量一致
            results.append({
                'origin_prompt': query if 'query' in locals() else '',
                'gold': gold if 'gold' in locals() else '',
                'prediction': '[]'  # 空列表
            })
            continue
    
    results_df = pd.DataFrame(results)
    
    # 验证结果数量
    if len(results_df) != total_samples:
        logging.error(f"结果数量 ({len(results_df)}) 与评估集数量 ({total_samples}) 不一致!")
    else:
        logging.info(f"成功生成 {len(results_df)} 条评估结果")
    
    results_df.to_csv('Conan-embedding-v1.csv', index=False)
    logging.info("评估结果已保存到 Conan-embedding-v1.csv")

def main():
    """主函数"""
    try:
        # 加载模型
        tokenizer, model, device = load_model()
        
        # 构建向量索引
        drug_files = ['drug1.csv']
        index, vectors, all_texts, description_texts = build_nearest_neighbors_index(drug_files, tokenizer, model, device)
        
        # 评估模型
        evaluate_model('eval.csv', index, vectors, all_texts, description_texts, tokenizer, model, device)
        
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
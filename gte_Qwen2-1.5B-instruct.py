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
    """加载预训练的gte_Qwen2-1.5B-instruct模型和tokenizer"""
    try:
        logging.info("开始加载模型...")
        model_name = "gte_Qwen2-1.5B-instruct"  # 改为本地gte_Qwen2-1.5B-instruct模型路径
        logging.info(f"尝试加载模型: {model_name}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer加载成功")
        
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        logging.info(f"Model已加载到{device}")
            
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误详情: {str(e)}")
        raise

def text_to_vector(text: str, tokenizer, model, device) -> np.ndarray:
    """将文本转换为向量"""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=32768, truncation=True, padding=True)
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
                         max_length=32768,  # 修改为gte_Qwen2-1.5B-instruct模型的max_length
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
    """
    构建最近邻索引
    Args:
        drug_files: drug文件路径列表
        tokenizer: 分词器
        model: 模型
        device: 设备
    Returns:
        Tuple[NearestNeighbors, np.ndarray, List[str], List[str]]: 索引、向量、文本列表、描述文本列表
    """
    try:
        all_texts = []
        description_texts = []
        
        # 读取所有drug文件
        for file_path in drug_files:
            try:
                # 使用header=None来读取所有行，不把第一行作为列名
                df = pd.read_csv(file_path, header=None)
                if len(df) > 0:
                    texts = df.iloc[:, 0].tolist()  # 第一列为文本
                    descriptions = df.iloc[:, 1].tolist()  # 第二列为描述
                    all_texts.extend(texts)
                    description_texts.extend(descriptions)
                    logging.info(f"成功读取{file_path}, 获取{len(texts)}条记录")
                else:
                    logging.warning(f"{file_path}为空文件")
            except Exception as e:
                logging.error(f"读取{file_path}失败: {str(e)}")
                continue
        
        if not all_texts:
            raise ValueError("没有读取到任何文本数据")
            
        # 将文本转换为向量
        vectors = process_batch_texts(all_texts, tokenizer, model, device)
        
        # 构建最近邻索引
        index = NearestNeighbors(n_neighbors=min(50, len(vectors)), metric='cosine')
        index.fit(vectors)
        
        logging.info(f"成功构建向量索引,共{len(vectors)}条记录")
        
        return index, vectors, all_texts, description_texts
        
    except Exception as e:
        logging.error(f"构建向量索引失败: {str(e)}")
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

def extract_drug_name(text):
    """
    从文本中提取药品名称
    只提取第一个[""药品名:XXX\n之后的内容，到\n为止
    """
    try:
        # 移除开头的[""
        if text.startswith('[""'):
            text = text[3:]
            
        # 查找"药品名:"的位置
        name_start = text.find("药品名:")
        if name_start == -1:
            return ""
            
        # 从"药品名:"后开始
        name_start += 4
        
        # 查找第一个换行符
        name_end = text.find("\n", name_start)
        if name_end == -1:
            return text[name_start:]
            
        return text[name_start:name_end]
        
    except Exception as e:
        logging.error(f"提取药品名称时出错: {str(e)}")
        return ""

def evaluate_model(eval_file, index, vectors, all_texts, description_texts, tokenizer, model, device):
    """评估模型性能"""
    try:
        # 读取评估数据集
        eval_df = pd.read_csv(eval_file)
        total_samples = len(eval_df)
        logging.info(f"加载评估集: {total_samples} 条数据")

        results = []
        for _, row in tqdm(eval_df.iterrows(), total=total_samples):
            query = row['instruction'] if pd.notna(row['instruction']) else row['input']
            
            # 获取相似文本
            similar_texts = get_similar_texts(query, index, vectors, all_texts, description_texts, tokenizer, model, device)
            
            # 只提取第一个药品名称
            drug_names = [extract_drug_name(text) for text in similar_texts]
            # 过滤掉空字符串
            drug_names = [name for name in drug_names if name]
            
            results.append({
                'origin_prompt': query,
                'gold': row['rel_drug'] if pd.notna(row['rel_drug']) else '',
                'prediction': drug_names
            })

        # 转换为DataFrame并保存
        results_df = pd.DataFrame(results)
        
        # 验证结果数量
        if len(results_df) != total_samples:
            logging.error(f"结果数量 ({len(results_df)}) 与评估集数量 ({total_samples}) 不一致!")
        else:
            logging.info(f"成功生成 {len(results_df)} 条评估结果")
        
        results_df.to_csv('gte_Qwen2-1.5B-instruct.csv', index=False)
        logging.info("评估结果已保存到 gte_Qwen2-1.5B-instruct.csv")

    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}")
        raise

def get_similar_texts(query: str, index: NearestNeighbors, vectors: np.ndarray, all_texts: List[str], description_texts: List[str], tokenizer, model, device) -> List[str]:
    """
    获取与查询最相似的文本
    Args:
        query: 查询文本
        index: NearestNeighbors索引
        vectors: 所有文本的向量
        all_texts: 所有原始文本
        description_texts: 所有描述文本
        tokenizer: 分词器
        model: 模型
        device: 设备
    Returns:
        List[str]: 相似文本列表
    """
    try:
        # 将查询转换为向量
        query_vector = text_to_vector(query, tokenizer, model, device)
        
        # 获取最近的50个向量的距离和索引
        distances, indices = index.kneighbors([query_vector], n_neighbors=min(50, len(all_texts)))
        
        # 获取对应的文本
        similar_texts = [all_texts[idx] for idx in indices[0]]
        
        return similar_texts
        
    except Exception as e:
        logging.error(f"获取相似文本失败: {str(e)}")
        return []

def main():
    """主函数"""
    try:
        # 加载模型
        tokenizer, model, device = load_model()
        
        # 构建drug文件列表
        drug_files = [f'drug/drug{i}.csv' for i in range(1, 51)]
        
        # 构建向量索引
        index, vectors, all_texts, description_texts = build_nearest_neighbors_index(drug_files, tokenizer, model, device)
        
        # 评估模型
        evaluate_model('eval.csv', index, vectors, all_texts, description_texts, tokenizer, model, device)
        
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
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
import gc

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
    """加载预训练的Chuxin-Embedding模型和tokenizer"""
    try:
        logging.info("开始加载模型...")
        model_name = "Chuxin-Embedding"
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
        raise

def load_rerank_model():
    """加载重排模型Conan-embedding-v1"""
    try:
        logging.info("开始加载重排模型...")
        model_name = "Conan-embedding-v1"
        logging.info(f"尝试加载重排模型: {model_name}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logging.info("重排Tokenizer加载成功")
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
        model.eval()
        logging.info(f"重排Model已加载到{device}")
            
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"加载重排模型失败: {str(e)}")
        raise

def text_to_vector(text: str, tokenizer, model, device) -> np.ndarray:
    """将文本转换为向量"""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=8194, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            vector = vector.astype('float32')
            vector = vector / np.linalg.norm(vector)
            
        return vector
    except Exception as e:
        logging.error(f"文本转向量失败: {str(e)}")
        raise

def compute_similarity(query: str, texts: List[str], tokenizer, model, device) -> List[float]:
    """计算查询和文本列表之间的相似度"""
    try:
        # 将查询转换为向量
        query_inputs = tokenizer(query, 
                               return_tensors="pt", 
                               max_length=512,
                               truncation=True, 
                               padding=True)
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            query_vector = query_outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 批量处理文本向量
        similarities = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:min(i + batch_size, len(texts))]
            batch_inputs = tokenizer(batch_texts, 
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True, 
                                   padding=True)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                batch_outputs = model(**batch_inputs)
                batch_vectors = batch_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                
                # 计算余弦相似度
                batch_similarities = np.dot(batch_vectors, query_vector)
                similarities.extend(batch_similarities)
                
            # 清理内存
            del batch_outputs
            del batch_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return similarities
    except Exception as e:
        logging.error(f"计算相似度失败: {str(e)}")
        raise

def rerank_texts(query: str, texts: List[str], tokenizer, model, device) -> List[str]:
    """使用重排模型对文本进行重新排序"""
    try:
        # 计算相似度
        similarities = compute_similarity(query, texts, tokenizer, model, device)
        
        # 将文本和相似度打包并排序
        text_scores = list(zip(texts, similarities))
        text_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的前50个文本
        return [text for text, _ in text_scores[:50]]
    except Exception as e:
        logging.error(f"重排文本失败: {str(e)}")
        raise

def process_batch_texts(texts: List[str], tokenizer, model, device, batch_size=128) -> np.ndarray:
    """批量处理文本到向量"""
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts in batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, 
                         return_tensors="pt", 
                         max_length=8194,
                         truncation=True, 
                         padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            vectors.extend(batch_vectors)
            
        # 清理内存
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.array(vectors)

def build_nearest_neighbors_index(drug_files: List[str], tokenizer, model, device) -> Tuple[NearestNeighbors, np.ndarray, List[str], List[str]]:
    """构建最近邻索引"""
    try:
        all_texts = []
        description_texts = []
        
        for file_path in drug_files:
            try:
                df = pd.read_csv(file_path, header=None)
                if len(df) > 0:
                    texts = df.iloc[:, 0].tolist()
                    descriptions = df.iloc[:, 1].tolist()
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
            
        vectors = process_batch_texts(all_texts, tokenizer, model, device)
        
        index = NearestNeighbors(n_neighbors=min(100, len(vectors)), metric='cosine')
        index.fit(vectors)
        
        logging.info(f"成功构建向量索引,共{len(vectors)}条记录")
        
        return index, vectors, all_texts, description_texts
        
    except Exception as e:
        logging.error(f"构建向量索引失败: {str(e)}")
        raise

def extract_drug_name(text):
    """从文本中提取药品名称"""
    try:
        if text.startswith('[""'):
            text = text[3:]
            
        name_start = text.find("药品名:")
        if name_start == -1:
            return ""
            
        name_start += 4
        name_end = text.find("\n", name_start)
        if name_end == -1:
            return text[name_start:]
            
        return text[name_start:name_end]
        
    except Exception as e:
        logging.error(f"提取药品名称时出错: {str(e)}")
        return ""

def get_similar_texts(query: str, index: NearestNeighbors, vectors: np.ndarray, all_texts: List[str], description_texts: List[str], tokenizer, model, device) -> List[str]:
    """获取与查询最相似的文本"""
    try:
        query_vector = text_to_vector(query, tokenizer, model, device)
        distances, indices = index.kneighbors([query_vector], n_neighbors=min(100, len(all_texts)))
        similar_texts = [all_texts[idx] for idx in indices[0]]
        return similar_texts
        
    except Exception as e:
        logging.error(f"获取相似文本失败: {str(e)}")
        return []

def evaluate_model(eval_file, index, vectors, all_texts, description_texts, tokenizer, model, device):
    """评估模型性能"""
    try:
        # 加载重排模型
        rerank_tokenizer, rerank_model, _ = load_rerank_model()
        logging.info("重排模型加载成功")
        
        # 读取评估数据集
        eval_df = pd.read_csv(eval_file)
        total_samples = len(eval_df)
        logging.info(f"加载评估集: {total_samples} 条数据")

        results = []
        for _, row in tqdm(eval_df.iterrows(), total=total_samples):
            query = row['instruction'] if pd.notna(row['instruction']) else row['input']
            
            # 获取相似文本（召回100个）
            similar_texts = get_similar_texts(query, index, vectors, all_texts, description_texts, tokenizer, model, device)
            
            # 使用重排模型对文本进行重新排序，获取top50
            reranked_texts = rerank_texts(query, similar_texts, rerank_tokenizer, rerank_model, device)
            
            # 提取药品名称
            drug_names = [extract_drug_name(text) for text in reranked_texts]
            drug_names = [name for name in drug_names if name]
            
            results.append({
                'origin_prompt': query,
                'gold': row['rel_drug'] if pd.notna(row['rel_drug']) else '',
                'prediction': drug_names
            })
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_df = pd.DataFrame(results)
        
        if len(results_df) != total_samples:
            logging.error(f"结果数量 ({len(results_df)}) 与评估集数量 ({total_samples}) 不一致!")
        else:
            logging.info(f"成功生成 {len(results_df)} 条评估结果")
        
        results_df.to_csv('Chuxin-Embedding_conan.csv', index=False)
        logging.info("评估结果已保存到 Chuxin-Embedding_conan.csv")

    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}")
        raise

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
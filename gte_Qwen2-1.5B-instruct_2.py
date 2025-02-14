import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import logging
from typing import List, Tuple
import sys
import pickle
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
    """加载预训练的gte_Qwen2-1.5B-instruct模型和tokenizer"""
    try:
        logging.info("开始加载模型...")
        model_name = "gte_Qwen2-1.5B-instruct"
        logging.info(f"尝试加载模型: {model_name}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        # 设置GPU内存分配策略
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
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
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        inputs = tokenizer(text, return_tensors="pt", max_length=8192, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            vector = vector.astype('float32')
            # 标准化向量
            vector = vector / np.linalg.norm(vector)
            
        # 清理内存
        del outputs
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return vector
    except Exception as e:
        logging.error(f"文本转向量失败: {str(e)}")
        raise

def process_batch_texts(texts: List[str], tokenizer, model, device, batch_size=32) -> np.ndarray:
    """批量处理文本到向量"""
    vectors = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts in batches"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, 
                             return_tensors="pt", 
                             max_length=8192,
                             truncation=True, 
                             padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # 标准化
                batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                vectors.extend(batch_vectors)
                
            # 清理内存
            del outputs
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        return np.array(vectors)
    except Exception as e:
        logging.error(f"批量处理文本失败: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def build_faiss_index(drug_files: List[str], tokenizer, model, device) -> Tuple[faiss.Index, np.ndarray, List[str], List[str]]:
    """构建FAISS索引"""
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
        
        # 构建FAISS索引
        dimension = 1536  # gte_Qwen2-1.5B-instruct的向量维度
        index = faiss.IndexFlatL2(dimension)
        
        # 将向量添加到索引中
        index.add(vectors.astype('float32'))
        
        logging.info(f"成功构建FAISS索引,共{len(vectors)}条记录")
        
        return index, vectors, all_texts, description_texts
        
    except Exception as e:
        logging.error(f"构建FAISS索引失败: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def evaluate_model(eval_file, index, vectors, all_texts, description_texts, tokenizer, model, device):
    """评估模型性能"""
    try:
        eval_df = pd.read_csv(eval_file)
        total_samples = len(eval_df)
        logging.info(f"加载评估集: {total_samples} 条数据")

        results = []
        for _, row in tqdm(eval_df.iterrows(), total=total_samples):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                query = row['instruction'] if pd.notna(row['instruction']) else row['input']
                similar_texts = get_similar_texts(query, index, vectors, all_texts, description_texts, tokenizer, model, device)
                
                drug_names = [extract_drug_name(text) for text in similar_texts]
                drug_names = [name for name in drug_names if name]
                
                results.append({
                    'origin_prompt': query,
                    'gold': row['rel_drug'] if pd.notna(row['rel_drug']) else '',
                    'prediction': drug_names
                })
                
                if len(results) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logging.error(f"处理样本时出错: {str(e)}")
                continue

        results_df = pd.DataFrame(results)
        
        if len(results_df) != total_samples:
            logging.error(f"结果数量 ({len(results_df)}) 与评估集数量 ({total_samples}) 不一致!")
        else:
            logging.info(f"成功生成 {len(results_df)} 条评估结果")
        
        results_df.to_csv('gte_Qwen2-1.5B-instruct_2.csv', index=False)
        logging.info("评估结果已保存到 gte_Qwen2-1.5B-instruct_2.csv")

    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_similar_texts(query: str, index: faiss.Index, vectors: np.ndarray, all_texts: List[str], description_texts: List[str], tokenizer, model, device) -> List[str]:
    """获取与查询最相似的文本"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        query_vector = text_to_vector(query, tokenizer, model, device)
        
        # 使用FAISS搜索最近邻
        k = min(50, len(all_texts))  # 获取前50个最相似的结果
        distances, indices = index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        # 获取对应的文本
        similar_texts = [all_texts[idx] for idx in indices[0]]
        
        return similar_texts
        
    except Exception as e:
        logging.error(f"获取相似文本失败: {str(e)}")
        return []
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def main():
    """主函数"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
        tokenizer, model, device = load_model()
        drug_files = [f'drug/drug{i}.csv' for i in range(1, 51)]
        
        # 使用FAISS构建索引
        index, vectors, all_texts, description_texts = build_faiss_index(drug_files, tokenizer, model, device)
        
        # 评估模型
        evaluate_model('eval.csv', index, vectors, all_texts, description_texts, tokenizer, model, device)
        
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
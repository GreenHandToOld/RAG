import os

import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

class EmbeddingEvaluator:
    def __init__(self):
        """初始化评估器"""
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """加载预训练的Conan-embedding-v1模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Conan-embedding-v1")
            self.model = AutoModel.from_pretrained("Conan-embedding-v1")
            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """将文本转换为向量并进行L2归一化"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            vector = vector.astype('float32')
            faiss.normalize_L2(vector.reshape(1, -1))
            return vector
        except Exception as e:
            print(f"文本向量化失败: {str(e)}")
            raise
    
    def build_faiss_index(self, drug_files: List[str]) -> Tuple[faiss.IndexFlatIP, List[str], List[str]]:
        """构建FAISS向量数据库"""
        vector_dimension = 1792  # Conan-embedding-v1的向量维度
        index = faiss.IndexFlatL2(vector_dimension)
        
        all_texts = []
        description_texts = []
        batch_size = 24  # 批处理大小
        
        for file in drug_files:
            try:
                print(f"\n处理文件: {file}")
                df = pd.read_csv(file)
                combined_texts = df.iloc[:, 0] + " " + df.iloc[:, 1]
                description_texts.extend(df.iloc[:, 1].tolist())
                all_texts.extend(combined_texts.tolist())
                
                # 批量处理向量转换
                for i in tqdm(range(0, len(combined_texts), batch_size), desc=f"Processing {file}"):
                    batch_texts = combined_texts[i:i + batch_size]
                    batch_vectors = [self.text_to_vector(text) for text in batch_texts]
                    vectors = np.array(batch_vectors)
                    faiss.normalize_L2(vectors)
                    index.add(vectors)
                
                print(f"成功添加 {len(combined_texts)} 个向量到索引")
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue
        
        if not all_texts:
            raise ValueError("没有成功处理任何文本数据")
        
        return index, all_texts, description_texts
    
    def evaluate_model(self, eval_file: str, index: faiss.IndexFlatIP, 
                      description_texts: List[str]) -> None:
        """评估模型性能并保存结果"""
        try:
            results = []
            eval_df = pd.read_csv(eval_file)
            
            for _, row in tqdm(eval_df.iterrows(), desc="Evaluating"):
                query = row['instruction']
                gold = row['rel_drug']
                
                query_vector = self.text_to_vector(query)
                D, I = index.search(query_vector.reshape(1, -1), 20)
                predictions = [description_texts[i] for i in I[0]]
                
                results.append({
                    'origin_prompt': query,
                    'gold': gold,
                    'prediction': '\n'.join(predictions)
                })
            
            results_df = pd.DataFrame(results)
            results_df.to_csv('Conan-embedding-v1.csv', index=False, encoding='utf-8')
            print("评估结果已保存到 Conan-embedding-v1.csv")
            
        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        evaluator = EmbeddingEvaluator()
        evaluator.load_model()
        
        drug_files = ['drug1.csv', 'drug2.csv', 'drug3.csv', 'drug4.csv']
        index, all_texts, description_texts = evaluator.build_faiss_index(drug_files)
        evaluator.evaluate_model('eval.csv', index, description_texts)
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()

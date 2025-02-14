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
import gc
import dask.dataframe as dd  # 导入dask

# 设置更详细的日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class DrugModel:
    def __init__(self, model_name="gte_Qwen2-7B-instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self):
        try:
            logging.info("开始加载模型...")
            logging.info(f"尝试加载模型: {self.model_name}")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"使用设备: {self.device}")

            # 设置GPU内存分配策略
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True

            # 指定缓存目录
            cache_dir = "model_cache"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            logging.info("Tokenizer加载成功")

            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir).to(self.device)
            self.model.eval()
            logging.info(f"Model已加载到{self.device}")

        except Exception as e:
            logging.error(f"加载模型失败: {str(e)}")
            logging.error(f"错误类型: {type(e).__name__}")
            logging.error(f"错误详情: {str(e)}")
            raise

    def text_to_vector(self, text: str) -> np.ndarray:
        try:
            # 输入验证
            if not isinstance(text, str) or text == "":
                raise ValueError("输入文本不能为空或必须为字符串类型")

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            inputs = self.tokenizer(text, return_tensors="pt", max_length=32768, truncation=True, padding=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 直接在GPU上计算向量
                vector = outputs.last_hidden_state[:, 0, :].to(torch.float32)
                # 在GPU上进行向量标准化
                vector = vector / torch.norm(vector, dim=1, keepdim=True)
                # 将结果复制到CPU
                vector = vector.cpu().numpy()[0]

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

    def process_batch_texts(self, texts: List[str], batch_size=32) -> np.ndarray:
        vectors = []
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts in batches"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_texts = texts[i:i + batch_size]
                # 批量tokenize
                inputs = self.tokenizer(batch_texts,
                                        return_tensors="pt",
                                        max_length=32768,
                                        truncation=True,
                                        padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_vectors = outputs.last_hidden_state[:, 0, :].to(torch.float32)
                    # 使用torch.nn.functional.normalize进行标准化
                    batch_vectors = torch.nn.functional.normalize(batch_vectors, dim=1)
                    vectors.extend(batch_vectors.cpu().numpy())

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
            # 最终清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def build_nearest_neighbors_index(self, drug_files: List[str]) -> Tuple[NearestNeighbors, np.ndarray, List[str], List[str]]:
        try:
            all_texts = []
            description_texts = []

            # 读取所有drug文件
            for file_path in drug_files:
                try:
                    # 使用dask读取文件
                    df = dd.read_csv(file_path, header=None).compute()
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
            vectors = self.process_batch_texts(all_texts)

            # 构建最近邻索引
            index = NearestNeighbors(n_neighbors=min(50, len(vectors)), metric='cosine')
            index.fit(vectors)

            logging.info(f"成功构建向量索引,共{len(vectors)}条记录")

            return index, vectors, all_texts, description_texts
        except Exception as e:
            logging.error(f"构建向量索引失败: {str(e)}")
            raise
        finally:
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def format_drug_info(text: str) -> str:
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

    @staticmethod
    def extract_drug_name(text):
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

    def evaluate_model(self, eval_file, index, vectors, all_texts, description_texts, batch_size=10):
        try:
            # 读取评估数据集
            eval_df = pd.read_csv(eval_file)
            total_samples = len(eval_df)
            logging.info(f"加载评估集: {total_samples}条数据")

            results = []
            # 批量处理评估数据
            for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating samples"):
                batch_df = eval_df.iloc[i:i + batch_size]
                for _, row in batch_df.iterrows():
                    try:
                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        query = row['instruction'] if pd.notna(row['instruction']) else row['input']

                        # 获取相似文本
                        similar_texts = self.get_similar_texts(query, index, vectors, all_texts, description_texts)

                        # 只提取第一个药品名称
                        drug_names = [self.extract_drug_name(text) for text in similar_texts]
                        # 过滤掉空字符串
                        drug_names = [name for name in drug_names if name]

                        results.append({
                            'origin_prompt': query,
                            'gold': row['rel_drug'] if pd.notna(row['rel_drug']) else '',
                            'prediction': drug_names
                        })

                        # 定期清理内存
                        if len(results) % 10 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    except Exception as e:
                        logging.error(f"处理样本时出错: {str(e)}")
                        continue

            # 转换为DataFrame并保存
            results_df = pd.DataFrame(results)

            # 验证结果数量
            if len(results_df)!= total_samples:
                logging.error(f"结果数量 ({len(results_df)})与评估集数量 ({total_samples})不一致!")
            else:
                logging.info(f"成功生成{len(results_df)}条评估结果")

            results_df.to_csv('gte_Qwen2-7B-instruct_2.csv', index=False)
            logging.info("评估结果已保存到gte_Qwen2-7B-instruct_2.csv")

        except Exception as e:
            logging.error(f"评估过程出错: {str(e)}")
            raise
        finally:
            # 最终清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_similar_texts(self, query: str, index: NearestNeighbors, vectors: np.ndarray, all_texts: List[str],
                          description_texts: List[str]) -> List[str]:
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 将查询转换为向量
            query_vector = self.text_to_vector(query)

            # 获取最近的50个向量的距离和索引
            distances, indices = index.kneighbors([query_vector], n_neighbors=min(50, len(all_texts)))

            # 获取对应的文本
            similar_texts = [all_texts[idx] for idx in indices[0]]

            return similar_texts
        except Exception as e:
            logging.error(f"获取相似文本失败: {str(e)}")
            return []
        finally:
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    try:
        # 创建DrugModel实例
        drug_model = DrugModel()
        drug_model.load_model()

        # 构建drug文件列表
        drug_files = [f'drug/drug{i}.csv' for i in range(1, 51)]

        # 构建向量索引
        index, vectors, all_texts, description_texts = drug_model.build_nearest_neighbors_index(drug_files)

        # 评估模型
        drug_model.evaluate_model('eval.csv', index, vectors, all_texts, description_texts)

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise
    finally:
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
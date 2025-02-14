import pandas as pd
import ast
from typing import Dict, Any

def convert_prediction_to_set(pred_str, top_k=10):
    """
    将prediction字符串转换为集合，并只取前top_k个元素
    """
    try:
        pred_list = ast.literal_eval(pred_str)
        return set(pred_list[:top_k])
    except:
        return set()

def convert_gold_to_set(gold_str):
    """
    将gold字符串转换为集合
    """
    try:
        return set(gold_str.split(','))
    except:
        return set()

def calculate_score(gold_set, pred_set):
    """
    计算得分
    """
    if not gold_set or not pred_set:
        return 0, 'no_match'
    
    if gold_set.issubset(pred_set):
        return 1, 'full_match'
    
    score = 0
    matched = False
    for gold_item in gold_set:
        if any(gold_item in pred_item for pred_item in pred_set):
            score += 1/len(gold_set)
            matched = True
    
    if matched:
        return score, 'partial_match'
    
    return 0, 'no_match'

def analyze_predictions_for_k(df: pd.DataFrame, k: int) -> Dict[str, Any]:
    """
    分析特定k值的预测结果
    """
    full_matches = 0
    partial_matches = 0
    no_matches = 0
    total_score = 0
    total_cases = len(df)
    
    for _, row in df.iterrows():
        pred_set = convert_prediction_to_set(row['prediction'], k)
        gold_set = convert_gold_to_set(row['gold'])
        
        score, match_type = calculate_score(gold_set, pred_set)
        
        if match_type == 'full_match':
            full_matches += 1
        elif match_type == 'partial_match':
            partial_matches += 1
        else:
            no_matches += 1
            
        total_score += score
    
    recall = total_score / total_cases if total_cases > 0 else 0
    
    return {
        'top_k': k,
        'full_matches': full_matches,
        'partial_matches': partial_matches,
        'no_matches': no_matches,
        'total_cases': total_cases,
        'recall': round(recall, 4)
    }

def analyze_all_k(file_path: str, max_k: int = 50) -> Dict[int, Dict[str, Any]]:
    """
    分析从1到max_k的所有k值的预测结果
    """
    df = pd.read_csv(file_path)
    results = {}
    
    for k in range(1, max_k + 1):
        results[k] = analyze_predictions_for_k(df, k)
    
    return results

if __name__ == "__main__":
    file_path = "Chuxin-Embedding.csv"
    results = analyze_all_k(file_path)
    
    # 打印所有结果
    print("各个top_k的评估结果:")
    print("-" * 80)
    print(f"{'top_k':^6} | {'完全匹配':^8} | {'部分匹配':^8} | {'无匹配':^8} | {'总样本数':^8} | {'Recall':^8}")
    print("-" * 80)
    
    for k, result in results.items():
        print(f"{k:^6} | {result['full_matches']:^8} | {result['partial_matches']:^8} | "
              f"{result['no_matches']:^8} | {result['total_cases']:^8} | {result['recall']:^8.4f}")
    
    # 将结果保存到CSV文件
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv('evaluation_results.csv')
    print("\n评估结果已保存到 evaluation_results.csv")

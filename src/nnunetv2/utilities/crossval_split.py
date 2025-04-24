# from typing import List

# import numpy as np
# from sklearn.model_selection import KFold


# def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
#     splits = []
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
#         train_keys = np.array(train_identifiers)[train_idx]
#         test_keys = np.array(train_identifiers)[test_idx]
#         splits.append({})
#         splits[-1]['train'] = list(train_keys)
#         splits[-1]['val'] = list(test_keys)
#     return splits


from typing import List, Dict
import numpy as np
import re
from sklearn.model_selection import KFold
import os
from batchgenerators.utilities.file_and_folder_operations import *

def extract_patient_id(filename: str) -> str:
    """
    ファイル名から患者IDを抽出する
    例: '6375522_20_R_FLOOR-3_0000.png' -> '6375522'
    """
    # ファイル名の最初の部分（アンダースコアまで）を患者IDとして抽出
    match = re.match(r'^(\d+)_', os.path.basename(filename))
    if match:
        return match.group(1)
    else:
        # IDが見つからない場合はファイル名自体を返す
        return os.path.basename(filename)

def generate_crossval_split(
    train_identifiers: List[str], 
    seed=12345, 
    n_splits=5
) -> List[Dict[str, List[str]]]:
    """
    患者IDに基づいてクロスバリデーション分割を生成する
    
    Parameters:
    -----------
    train_identifiers: 訓練用の画像IDのリスト
    seed: 乱数シード
    n_splits: 分割数
    
    Returns:
    --------
    分割情報のリスト。各要素は辞書で、'train'と'val'キーを持つ
    """
    # 各画像IDから患者IDを抽出
    patient_ids = [extract_patient_id(identifier) for identifier in train_identifiers]
    
    # 画像IDと患者IDのマッピングを作成
    id_to_patient = {identifier: patient_id for identifier, patient_id in zip(train_identifiers, patient_ids)}
    
    # ユニークな患者IDのリストを取得
    unique_patient_ids = list(set(patient_ids))
    
    # 患者IDでKFold分割を行う
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for i, (train_idx, val_idx) in enumerate(kfold.split(unique_patient_ids)):
        # 訓練・検証用の患者IDを取得
        train_patient_ids = [unique_patient_ids[idx] for idx in train_idx]
        val_patient_ids = [unique_patient_ids[idx] for idx in val_idx]
        
        # 各患者IDに対応する画像IDを取得
        train_keys = [identifier for identifier in train_identifiers 
                     if id_to_patient[identifier] in train_patient_ids]
        val_keys = [identifier for identifier in train_identifiers 
                   if id_to_patient[identifier] in val_patient_ids]
        
        # 分割情報を保存
        splits.append({})
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = val_keys
    
    return splits
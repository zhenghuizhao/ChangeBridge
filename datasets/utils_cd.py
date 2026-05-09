import os

# your_dataset_path/
# ├─A/          # 所有参考图像
# ├─B/          # 所有真实图像
# ├─label/      # 可能包含标签信息（具体用途视项目而定）
# └─list/
#     ├─train.txt  # 训练集图像文件名列表
#     ├─val.txt    # 验证集图像文件名列表
#     └─test.txt   # 测试集图像文件名列表


import os
import logging


def get_image_triplets_from_list(list_dir, split, A_dir, B_dir, label_dir):
    """
    根据指定的数据集分割（train, val, test）从list文件中读取图像对和标签的路径。

    Args:
        list_dir (str): 存放list文件的目录路径。
        split (str): 数据集分割名称，必须是 'train', 'val' 或 'test'。
        A_dir (str): 参考图像A的目录路径。
        B_dir (str): 真实图像B的目录路径。
        label_dir (str): 标签文件的目录路径。

    Returns:
        list of tuples: 每个元组包含一个参考图像A的路径、对应的真实图像B的路径和标签路径。
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    list_file = os.path.join(list_dir, f"{split}.txt")
    image_triplets = []

    if not os.path.exists(list_file):
        logger.error(f"List file {list_file} does not exist.")
        raise FileNotFoundError(f"List file {list_file} does not exist.")

    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        filename = line.strip()
        if not filename:
            continue  # 跳过空行

        A_path = os.path.abspath(os.path.join(A_dir, filename))
        B_path = os.path.abspath(os.path.join(B_dir, filename))
        label_path = os.path.abspath(os.path.join(label_dir, filename))  # 假设标签文件名与图像文件名相同

        missing = []
        if not os.path.exists(A_path):
            missing.append('A')
        if not os.path.exists(B_path):
            missing.append('B')
        if not os.path.exists(label_path):
            missing.append('label')

        if not missing:
            image_triplets.append((A_path, B_path, label_path))
        else:
            logger.warning(f"Missing {', '.join(missing)} image(s) for {filename}")

    return image_triplets



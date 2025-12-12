import csv
import torch
from typing import Any, List, Tuple


def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X: torch.Tensor of dtype=torch.object
           each element is {"text": title_str}
        y: torch.Tensor of dtype=torch.long
           each element is int(0/1)
    """

    X_list = []
    y_list = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"].strip()
            label = int(row["label"])
            # TODO: 数据清洗 这里没做
            # model.predict expects dict {"text": ...}
            X_list.append({"text": title})
            y_list.append(label)

    # Convert to torch Tensors (object dtype allowed)
    X_tensor = torch.tensor(X_list, dtype=torch.object)
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    return X_tensor, y_tensor

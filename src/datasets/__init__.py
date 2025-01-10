from .stir import stir

ALL = [
    "STIR"
]


def load(dataset_name: str):
    dataset_name = dataset_name.upper()
    assert dataset_name in ALL
    if dataset_name == "STIR":
        input_path, bbox_start, ground_truth = stir.load_data()
    else:
        print(f"Unknown dataset {dataset_name} selected.")
        return
    return input_path, bbox_start, ground_truth


import os
from typing import Set, Tuple
import pandas as pd

def read_param_codes_from_orc(orc_path: str) -> Set[str]:
    """
    读取指定 ORC 文件的 param_code 列，并返回唯一值集合。
    仅从单个文件读取（xw_data_00.orc）。
    """
    if not os.path.isfile(orc_path):
        raise FileNotFoundError(f"找不到 ORC 文件: {orc_path}")

    try:
        df = pd.read_orc(orc_path, columns=["param_code"])
    except TypeError:
        df = pd.read_orc(orc_path)
        if "param_code" not in df.columns:
            raise KeyError(f"{orc_path} 中没有 'param_code' 列")
        df = df[["param_code"]]

    return set(df["param_code"].dropna().astype(str).unique().tolist())


def compare_three_orc_param_codes_only_xw(
    dir1: str, dir2: str, dir3: str, out_dir: str = ".",
    file_name: str = "xw_data_00.orc"
) -> Tuple[pd.DataFrame, Set[str], Set[str], Set[str]]:
    """
    只比较每个目录下的 file_name（默认 xw_data_00.orc）。
    产出：
      - param_code_comparison.csv：汇总（布尔矩阵 + 分类）
      - only_in_1.txt / only_in_2.txt / only_in_3.txt：仅在对应目录出现的参数清单
    """
    os.makedirs(out_dir, exist_ok=True)

    f1 = os.path.join(dir1, file_name)
    f2 = os.path.join(dir2, file_name)
    f3 = os.path.join(dir3, file_name)

    s1 = read_param_codes_from_orc(f1)
    s2 = read_param_codes_from_orc(f2)
    s3 = read_param_codes_from_orc(f3)

    only1 = s1 - (s2 | s3)
    only2 = s2 - (s1 | s3)
    only3 = s3 - (s1 | s2)

    all_codes = sorted(s1 | s2 | s3)
    df = pd.DataFrame({
        "param_code": all_codes,
        "in_dir1": [c in s1 for c in all_codes],
        "in_dir2": [c in s2 for c in all_codes],
        "in_dir3": [c in s3 for c in all_codes],
    })

    def label_row(r):
        flags = (r["in_dir1"], r["in_dir2"], r["in_dir3"])
        if flags == (True, False, False):
            return "only_dir1"
        if flags == (False, True, False):
            return "only_dir2"
        if flags == (False, False, True):
            return "only_dir3"
        return "multiple"

    df["category"] = df.apply(label_row, axis=1)

    # 输出文件
    summary_csv = os.path.join(out_dir, "param_code_comparison.csv")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "only_in_1.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(only1)))
    with open(os.path.join(out_dir, "only_in_2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(only2)))
    with open(os.path.join(out_dir, "only_in_3.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(only3)))

    print(f"[done] 汇总: {summary_csv}")
    print(f"目录1 独有: {len(only1)}，目录2 独有: {len(only2)}，目录3 独有: {len(only3)}，总唯一参数: {len(all_codes)}")
    return df, only1, only2, only3


# ===== 使用示例 =====
if __name__ == "__main__":
    dir1 = "./CSCN-A0007/2025-01-15"
    dir2 = "./CSCN-A0007/2025-03-23"
    dir3 = "./CSCN-A0007/2025-06-29"
    out_dir = "./param_code_diff_outputs"

    compare_three_orc_param_codes_only_xw(dir1, dir2, dir3, out_dir)

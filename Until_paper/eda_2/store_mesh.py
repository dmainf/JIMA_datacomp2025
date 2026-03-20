import os
import pandas as pd
import chardet
import datetime as dt

# === ディレクトリ構成 ===
BASE_DIR = os.path.dirname(__file__)  # ← eda_2フォルダが基準
STORE_FILE = os.path.join(BASE_DIR, "data", "Store_4th.csv")  # ← eda_2/data/Store_4th.csv
STORE_CODE_FILE = os.path.join(BASE_DIR, "..", "data", "store_code.txt")  # ← launch/data/store_code.txt
DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")  # ← eda_2/data_store/


# === 共通関数（3次/4次メッシュ対応） ===
def get_population_from_mesh(mesh_code: str, mesh_type="3rd"):
    """
    mesh_type : "3rd" or "4th"
    3次メッシュ → tblT001140S~
    4次メッシュ → tblT001141H~
    """
    try:
        mesh_code = str(mesh_code).strip().replace("　", "")
        if not mesh_code.isdigit() or len(mesh_code) < 4:
            return 0

        prefix = mesh_code[:4]
        if mesh_type == "4th":
            file_name = f"tblT001141H{prefix}.csv"
        else:
            file_name = f"tblT001140S{prefix}.csv"

        file_path = os.path.join(DATA_STORE_DIR, file_name)

        if not os.path.exists(file_path):
            return 0

        with open(file_path, "rb") as f:
            raw = f.read(10000)
            enc = chardet.detect(raw)["encoding"] or "utf-8"

        try:
            df_sub = pd.read_csv(file_path, encoding=enc, header=None, low_memory=False)
        except UnicodeDecodeError:
            df_sub = pd.read_csv(file_path, encoding="cp932", header=None)

        df_sub[0] = df_sub[0].astype(str).str.strip().str.replace("　", "")
        df_match = df_sub[df_sub[0] == mesh_code]

        if not df_match.empty:
            val = df_match.iloc[0, 4]
            if isinstance(val, str):
                val = val.replace(",", "").strip()
            try:
                return float(val)
            except ValueError:
                return 0
        else:
            return 0

    except Exception as e:
        print(f"Error : mesh_code = {mesh_code} {e}")
        return 0


# === Store_4th.csv 読み込み ===
try:
    df_store = pd.read_csv(STORE_FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df_store = pd.read_csv(STORE_FILE, encoding="cp932")

# === store_code.txt ===
try:
    df_codes = pd.read_csv(
        STORE_CODE_FILE,
        sep=r"[\s　]+",
        engine="python",
        encoding="utf-8-sig"
    )
    df_codes.columns = df_codes.columns.str.strip().str.replace("　", "")
    if len(df_store) == len(df_codes):
        for col in ["書店コード", "書店名"]:
            if col in df_store.columns and col in df_codes.columns:
                df_store[col] = df_store[col].fillna(df_codes[col])
                df_store.loc[df_store[col] == "", col] = df_codes[col]
        print("input store_code.txt")
    else:
        print(f"Error : Store_4th.csv={len(df_store)}, store_code.txt={len(df_codes)}")
except Exception as e:
    print(f"Not Found : store_code.txt {e}")


"""
 N
W E
 S
"""
"""
value of mesh
1 2 3
4 5 6
7 8 9
"""

# === 3次メッシュ人口 ===
print("input mesh population...")
for i in range(1, 10):
    col = f"メッシュ{i}"
    new_col = f"メッシュ{i}_人口"
    if col in df_store.columns:
        df_store[new_col] = df_store[col].apply(lambda x: get_population_from_mesh(x, mesh_type="3rd"))

for col in [f"メッシュ{i}_人口" for i in range(1, 10)]:
    df_store[col] = pd.to_numeric(df_store[col], errors="coerce").fillna(0)

周辺メッシュ = [1, 2, 3, 4, 6, 7, 8, 9]
df_store["周辺合計人口"] = df_store[[f"メッシュ{i}_人口" for i in 周辺メッシュ]].sum(axis=1)
df_store["合計人口"] = df_store[[f"メッシュ{i}_人口" for i in range(1, 10)]].sum(axis=1)

# === 4次メッシュ人口 ===
print("input 4th mesh population...")
for i in range(1, 10):
    col = f"4th_メッシュ{i}"
    new_col = f"4th_メッシュ{i}_人口"
    if col in df_store.columns:
        df_store[new_col] = df_store[col].apply(lambda x: get_population_from_mesh(x, mesh_type="4th"))

for col in [f"4th_メッシュ{i}_人口" for i in range(1, 10)]:
    df_store[col] = pd.to_numeric(df_store[col], errors="coerce").fillna(0)

df_store["4th_周辺合計人口"] = df_store[[f"4th_メッシュ{i}_人口" for i in 周辺メッシュ]].sum(axis=1)
df_store["4th_合計人口"] = df_store[[f"4th_メッシュ{i}_人口" for i in range(1, 10)]].sum(axis=1)
print("input 4th mesh complete")

# === 営業時間 ===
time_cols = ["開店時間(平)", "閉店時間(平)", "開店時間(特)", "閉店時間(特)"]

def convert_time(value):
    try:
        value = int(value)
        hour = value // 100
        minute = value % 100
        if not (0 <= minute <= 59):
            return None
        if hour == 24:
            hour = 0
        if not (0 <= hour <= 23):
            return None
        return dt.time(hour, minute, 0)
    except Exception:
        return None

for col in time_cols:
    if col in df_store.columns:
        df_store[col] = df_store[col].apply(convert_time)

for mode in ["(平)", "(特)"]:
    open_col = f"開店時間{mode}"
    close_col = f"閉店時間{mode}"
    work_col = f"営業時間{mode}"

    if open_col in df_store.columns and close_col in df_store.columns:
        def calc_duration(row):
            open_t = row[open_col]
            close_t = row[close_col]
            if pd.notnull(open_t) and pd.notnull(close_t):
                start = dt.datetime.combine(dt.date.today(), open_t)
                end = dt.datetime.combine(dt.date.today(), close_t)
                if open_t == close_t:
                    return 1440  # 24時間営業
                if end <= start:
                    end += dt.timedelta(days=1)
                delta = (end - start).seconds
                return delta // 60
            return None
        df_store[work_col] = df_store.apply(calc_duration, axis=1)

print("change datetime")

# === 昼間/夜間割合 ===
if all(col in df_store.columns for col in ["市区別_夜間人口", "市区別_昼間人口"]):
    df_store["昼間/夜間割合"] = df_store.apply(
        lambda x: round((x["市区別_昼間人口"] / x["市区別_夜間人口"]) * 100, 1)
        if x["市区別_夜間人口"] not in [0, None, ""] else 0,
        axis=1
    ).astype(float)
    print("calculation day/night")
else:
    print("Not Found : daytime or nighttime")


# === 型変換 ===
expected_dtypes = {
    "書店コード": "int",
    "書店名": "object",
    "住所": "object",
    "駅距離": "int",
    "開店時間(平)": "datetime",
    "閉店時間(平)": "datetime",
    "営業時間(平)": "int",
    "開店時間(特)": "datetime",
    "閉店時間(特)": "datetime",
    "営業時間(特)": "int",
    "駅構内": "int",
    "複合施設": "int",
    "独立店舗": "int",
    "市区別_夜間人口": "int",
    "市区別_昼間人口": "int",
    "昼間/夜間割合": "float",
    "昼間フラグ": "int"
}

for i in range(1, 10):
    expected_dtypes[f"メッシュ{i}"] = "int"
    expected_dtypes[f"メッシュ{i}_人口"] = "int"
expected_dtypes["周辺合計人口"] = "int"
expected_dtypes["合計人口"] = "int"

for i in range(1, 10):
    expected_dtypes[f"4th_メッシュ{i}"] = "int"
    expected_dtypes[f"4th_メッシュ{i}_人口"] = "int"
expected_dtypes["4th_周辺合計人口"] = "int"
expected_dtypes["4th_合計人口"] = "int"

def fix_dtype(series, target_type):
    if target_type == "int":
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    elif target_type == "float":
        return pd.to_numeric(series, errors="coerce").astype(float)
    elif target_type == "datetime":
        if all(isinstance(v, dt.time) or pd.isnull(v) for v in series):
            return series
        return pd.to_datetime(series, errors="coerce").dt.time
    elif target_type == "object":
        return series.astype(str)
    return series

for col, t in expected_dtypes.items():
    if col in df_store.columns:
        df_store[col] = fix_dtype(df_store[col], t)
    else:
        print(f"Not Found : column {col}")

print("check complete")

# === 出力 ===
output_csv = os.path.join(BASE_DIR, "data", "Store_mesh.csv")
output_parquet = os.path.join(BASE_DIR, "data", "Store_mesh.parquet")

df_store.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"output CSV : {output_csv}")

df_store.to_parquet(output_parquet, index=False)
print(f"output Parquet : {output_parquet}")

print("Complete!")

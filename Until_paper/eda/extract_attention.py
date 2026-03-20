"""
GluonTS Temporal Fusion TransformerからAttention重みを抽出・可視化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset
from common import load_encoders
from transformer import PREDICTION_LENGTH, CONTEXT_LENGTH
from feature_eng import STATIC_CAT_COLS, STATIC_REAL_COLS, TEMPORAL_COLS, TIME_COLS

matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'YuGothic', 'Hiragino Maru Gothic Pro']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

print("Loading model and dataset...")
model_path = Path("model")
predictor = PyTorchPredictor.deserialize(model_path)
dataset_df = pd.read_parquet('dataset.parquet')
encoders = load_encoders('encoders.pkl')
print("Loaded!")

# モデルの内部構造を取得
model = predictor.prediction_net

# Attention重みと特徴量重要度を保存する辞書
attention_weights = {}
static_feature_weights = {}
past_feature_weights = {}
future_feature_weights = {}

def create_attention_hook(name):
    """MultiheadAttention層からのAttention Weightをキャプチャするフック"""
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            if attn is not None:
                attention_weights[name] = attn.detach().cpu().numpy()
    return hook

def create_feature_weight_hook(name, weight_dict):
    """VariableSelectionNetwork層からの特徴量重要度をキャプチャするフック"""
    def hook(module, input, output):
        # output[1]がsoftmax後の重み
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1]
            if weights is not None:
                weight_dict[name] = weights.detach().cpu().numpy()
    return hook

# フックを登録
hooks = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.MultiheadAttention):
        print(f"Registering attention hook on: {name}")
        hook = module.register_forward_hook(create_attention_hook(name))
        hooks.append(hook)
    elif name == 'model.static_selector':
        print(f"Registering static feature hook on: {name}")
        hook = module.register_forward_hook(create_feature_weight_hook(name, static_feature_weights))
        hooks.append(hook)
    elif name == 'model.ctx_selector':
        print(f"Registering past feature hook on: {name}")
        hook = module.register_forward_hook(create_feature_weight_hook(name, past_feature_weights))
        hooks.append(hook)
    elif name == 'model.tgt_selector':
        print(f"Registering future feature hook on: {name}")
        hook = module.register_forward_hook(create_feature_weight_hook(name, future_feature_weights))
        hooks.append(hook)


# === Extracting Weights ===
print("\n=== Extracting Attention and Feature Weights ===")

store_codes = list(range(1, 26)) + list(range(28, 36))
output_dir = Path("score/attention_and_features")
output_dir.mkdir(parents=True, exist_ok=True)

# VSNの実際の出力次元に基づいた特徴量名を取得
# モデルの実際の構造を確認
num_static_features = len(model.model.static_selector.variable_networks)
num_past_features = len(model.model.ctx_selector.variable_networks)
num_future_features = len(model.model.tgt_selector.variable_networks)

print(f"VSN dimensions: static={num_static_features}, past={num_past_features}, future={num_future_features}")

# 静的特徴量名（19 categorical + 1 real = 20）
static_feature_names = STATIC_CAT_COLS[:num_static_features] if len(STATIC_CAT_COLS) >= num_static_features else \
    STATIC_CAT_COLS + [f"static_real_{i}" for i in range(num_static_features - len(STATIC_CAT_COLS))]

# 過去の特徴量名（6個）
past_feature_names = [f"past_feat_{i}" for i in range(num_past_features)]

# 未来の特徴量名（4個）
future_feature_names = TEMPORAL_COLS[:num_future_features] if len(TEMPORAL_COLS) >= num_future_features else \
    TEMPORAL_COLS + [f"future_feat_{i}" for i in range(num_future_features - len(TEMPORAL_COLS))]

for store_code in store_codes:
    print(f"--- Processing Store {store_code} ---")
    store_file = Path(f"data/by_store/df_{store_code}.parquet")
    if not store_file.exists():
        print(f"  Skipped: File not found")
        continue
    
    df_store = pd.read_parquet(store_file)

    sales_by_book = df_store.groupby('書名')['売上'].agg(['sum', 'count'])
    sales_by_book = sales_by_book[sales_by_book['count'] >= CONTEXT_LENGTH + PREDICTION_LENGTH]
    sales_by_book = sales_by_book.sort_values('sum', ascending=False)

    if len(sales_by_book) == 0:
        print(f"  Skipped: No books with sufficient data in store {store_code}")
        continue

    top_books = sales_by_book.head(5).index.tolist()

    for book_name in top_books:
        print(f"  Analyzing book: {book_name}")

        try:
            encoded_book_name = encoders['書名'].transform([str(book_name)])[0]
            book_features = dataset_df[(dataset_df['書店コード'] == store_code) &
                                      (dataset_df['書名'] == encoded_book_name)].sort_values('日付')
            if book_features.empty:
                print(f"    Skipped: Book features not found in the main dataset.")
                continue

            df_book = df_store[df_store['書名'] == book_name].sort_values('日付')
            sales = df_book['売上'].values

            min_split = CONTEXT_LENGTH
            max_split = len(sales) - PREDICTION_LENGTH
            if max_split <= min_split:
                print(f"    Skipped book '{book_name}': insufficient data length.")
                continue
            split_point = min_split + (max_split - min_split) // 2

            start_date = pd.to_datetime(book_features['日付'].iloc[0])
            
            # --- ここからデータ準備 ---
            static_cat = [int(book_features[col].iloc[0]) for col in STATIC_CAT_COLS]
            
            # `static_real` の準備を修正
            static_real_cols_present = [c for c in STATIC_REAL_COLS if c in book_features.columns]
            static_real = [float(book_features[col].iloc[0]) for col in static_real_cols_present]
            
            # TIME_COLSからの特徴量追加
            for time_col in TIME_COLS:
                if time_col in book_features.columns:
                    time_val = book_features[time_col].iloc[0]
                    # 元のコードのロジックを再現するが、datetime型でない場合を考慮
                    if isinstance(time_val, pd.Timestamp):
                        static_real.append(float(time_val.hour + time_val.minute / 60.0))
                    else:
                        static_real.append(0.0) # 存在しないか、型が違う場合は0を追加
                else:
                    static_real.append(0.0)

            test_end = split_point + PREDICTION_LENGTH
            feat_dynamic_real = book_features[TEMPORAL_COLS].values[:test_end].T.astype(np.float32)

            lag_cols = sorted([c for c in book_features.columns if c.startswith('売上_lag')], key=lambda x: int(x.split('lag')[1]))
            roll_mean_cols = sorted([c for c in book_features.columns if 'roll_mean' in c], key=lambda x: int(x.split('mean')[1]))
            roll_std_cols = sorted([c for c in book_features.columns if 'roll_std' in c], key=lambda x: int(x.split('std')[1]))
            roll_max_cols = sorted([c for c in book_features.columns if 'roll_max' in c], key=lambda x: int(x.split('max')[1]))
            past_dynamic_cols = lag_cols + roll_mean_cols + roll_std_cols + roll_max_cols
            feat_dynamic_past = book_features[past_dynamic_cols].values[:split_point].T.astype(np.float32)

            test_entry = {
                "start": pd.Period(start_date, freq="D"),
                "target": sales[:split_point],
                "feat_static_cat": static_cat,
                "feat_static_real": static_real,
                "feat_dynamic_real": feat_dynamic_real,
                "past_feat_dynamic_real": feat_dynamic_past
            }
            # --- データ準備ここまで ---

            print("    Running prediction to capture weights...")
            test_dataset = ListDataset([test_entry], freq="D")
            forecast_it = predictor.predict(test_dataset)
            forecast = list(forecast_it)[0]

            # --- 可視化 ---
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Store {store_code} - {book_name[:40]}\nAttention and Feature Importance Analysis', fontsize=18, fontweight='bold')
            axes = axes.flatten()

            # 1. Temporal Attention
            ax = axes[0]
            if attention_weights:
                name, weights = list(attention_weights.items())[0]
                attn = weights[0]
                im = ax.imshow(attn, cmap='viridis', aspect='auto')
                ax.set_xlabel('Key Position (Past Context)', fontsize=12)
                ax.set_ylabel('Query Position (Future Prediction)', fontsize=12)
                ax.set_title('Temporal Attention Pattern', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax, label='Attention Weight')
                ax.grid(False)
            else:
                ax.text(0.5, 0.5, "Not captured", ha='center', va='center')
                ax.axis('off')

            # 2. Static Feature Importance
            ax = axes[1]
            if static_feature_weights:
                name, weights = list(static_feature_weights.items())[0]
                weights_flat = weights.flatten()
                if len(weights_flat) == len(static_feature_names):
                    df_static = pd.DataFrame({'feature': static_feature_names, 'importance': weights_flat}).sort_values('importance', ascending=False)
                    ax.barh(df_static['feature'], df_static['importance'], color='skyblue')
                    ax.set_xlabel('Importance Weight', fontsize=12)
                    ax.set_title('Static Feature Importance', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, f"Mismatch: {len(weights_flat)} vs {len(static_feature_names)}", ha='center', va='center', wrap=True)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Not captured", ha='center', va='center')
                ax.axis('off')

            # 3. Past Temporal Feature Importance
            ax = axes[2]
            if past_feature_weights:
                name, weights = list(past_feature_weights.items())[0]
                weights_mean = weights.mean(axis=1).flatten()
                if len(weights_mean) == len(past_feature_names):
                    df_past = pd.DataFrame({'feature': past_feature_names, 'importance': weights_mean}).sort_values('importance', ascending=False)
                    ax.barh(df_past['feature'], df_past['importance'], color='lightgreen')
                    ax.set_xlabel('Importance Weight', fontsize=12)
                    ax.set_title('Past Temporal Feature Importance', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, f"Mismatch: {len(weights_mean)} vs {len(past_feature_names)}", ha='center', va='center', wrap=True)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Not captured", ha='center', va='center')
                ax.axis('off')

            # 4. Future Temporal Feature Importance
            ax = axes[3]
            if future_feature_weights:
                name, weights = list(future_feature_weights.items())[0]
                weights_mean = weights.mean(axis=1).flatten()
                if len(weights_mean) == len(future_feature_names):
                    df_future = pd.DataFrame({'feature': future_feature_names, 'importance': weights_mean}).sort_values('importance', ascending=False)
                    ax.barh(df_future['feature'], df_future['importance'], color='salmon')
                    ax.set_xlabel('Importance Weight', fontsize=12)
                    ax.set_title('Future Temporal Feature Importance', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, f"Mismatch: {len(weights_mean)} vs {len(future_feature_names)}", ha='center', va='center', wrap=True)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Not captured", ha='center', va='center')
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            safe_book_name = "".join(c for c in book_name if c.isalnum())[:20]
            output_file = output_dir / f'store_{store_code}_{safe_book_name}_full_analysis.png'
            plt.savefig(output_file, dpi=150)
            print(f"    Saved analysis plot: {output_file}")
            plt.close()

            # Clear dictionaries for next loop
            attention_weights.clear()
            static_feature_weights.clear()
            past_feature_weights.clear()
            future_feature_weights.clear()

        except Exception as e:
            import traceback
            print(f"    Error processing book '{book_name}' in store {store_code}: {e}")
            traceback.print_exc()
            # Clear dicts on error too
            attention_weights.clear()
            static_feature_weights.clear()
            past_feature_weights.clear()
            future_feature_weights.clear()


# フックを削除
for hook in hooks:
    hook.remove()

print("\n\nExtraction and visualization complete.")

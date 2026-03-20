import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
from typing import Dict, List

# --- GluonTS ---
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset
# 以下の2行をインポートセクション（15行目あたり）に追加
from gluonts.itertools import Map
import torch.utils.data
from gluonts.itertools import Map
import torch.utils.data

# --- o_transformer.py / o_feature_eng.py から必要な要素をインポート ---
from o_feature_eng import (
    create_slices, 
    process_group, 
    STATIC_CAT_COLS, 
    STATIC_REAL_COLS, 
    TIME_COLS,
    TEMPORAL_COLS,
    get_feature_windows
)
from o_transformer import (
    CONTEXT_LENGTH, 
    PREDICTION_LENGTH, 
    REGION_EXTEND
)

# ----------------------------------------------------------------------
# 警告の非表示
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 日本語フォントの設定
try:
    # Mac/Linux/Windowsで利用可能なフォントを探す
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    font_family = "Hiragino Sans" if "Hiragino Sans" in available_fonts else \
                  "Yu Gothic" if "Yu Gothic" in available_fonts else \
                  "MS Gothic" if "MS Gothic" in available_fonts else \
                  "Noto Sans CJK JP" if "Noto Sans CJK JP" in available_fonts else \
                  "sans-serif" # デフォルト
    
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Using font: {font_family}")
except Exception as e:
    print(f"Font setup failed: {e}. Using default sans-serif.")

# ----------------------------------------------------------------------

# グローバル変数（フックでキャプチャしたアテンションを保存するため）
attention_weights = {}

def get_attention_hook(module_name: str):
    """
    指定されたモジュールのアテンション・ウェイト（出力）をキャプチャするフック関数を生成する
    """
    def hook(model, input, output):
        try:
            # weight_networkの出力（変数選択ウェイト）
            if 'weight' in module_name:
                # output is the attention weights tensor
                if torch.is_tensor(output):
                    attention_weights[module_name] = output.detach().cpu().numpy()
                elif isinstance(output, tuple):
                    # If tuple, take first element
                    attention_weights[module_name] = output[0].detach().cpu().numpy()
                else:
                    print(f"Warning: Unexpected output format for {module_name}: {type(output)}")

            # MultiheadAttentionの出力 (attn_output, attn_weights)
            elif 'self_attention' in module_name:
                if isinstance(output, tuple) and len(output) >= 2:
                    # output[1] is attention weights
                    attention_weights[module_name] = output[1].detach().cpu().numpy()
                else:
                    print(f"Warning: Unexpected output format for {module_name}")
        except Exception as e:
            print(f"Error in hook {module_name}: {e}")

    return hook

def get_feature_names(dataset: pd.DataFrame, model_inputs: Dict) -> Dict[str, List[str]]:
    """
    実際のモデル入力形状に基づいて特徴量リストを構築する
    """

    # 1. 静的カテゴリ (Static Categorical)
    static_cat = STATIC_CAT_COLS  # 7個

    # 2. 静的実数値 (Static Real)
    # モデル入力のfeat_static_realの次元数を確認
    # 実際には1次元しかないようなので、それに合わせる
    num_static_real = model_inputs["feat_static_real"].shape[1] if "feat_static_real" in model_inputs else 1
    static_real = STATIC_REAL_COLS[:num_static_real]  # 実際の数に合わせる

    # 3. 将来の動的 (Decoder / Future Dynamic)
    # feat_dynamic_realの最後の次元が特徴量数
    num_decoder = model_inputs["feat_dynamic_real"].shape[2] if "feat_dynamic_real" in model_inputs else len(TEMPORAL_COLS)
    decoder_dynamic = TEMPORAL_COLS[:num_decoder]

    # 4. 過去の動的 (Encoder / Past Dynamic)
    # TFTモデルはctx_selector（エンコーダー）で過去の特徴量を処理する
    # past_feat_dynamic_realがない場合、TFTは過去のターゲット値のみを使用
    # しかし、encoder_vsn_weightの形状から実際の特徴量数を推測
    if "past_feat_dynamic_real" in model_inputs:
        num_encoder = model_inputs["past_feat_dynamic_real"].shape[2]
        # Roll特徴量から実際の数に合わせる
        roll_cols = sorted(
            [c for c in dataset.columns if 'POS販売冊数_roll' in c],
            key=lambda x: (
                int(x.split('roll_')[1].split('mean')[-1].split('std')[-1].split('max')[-1]),
                x.split('roll_')[1].split('mean')[0].split('std')[0].split('max')[0]
            )
        )
        encoder_dynamic = roll_cols[:num_encoder]
    else:
        # TFTがターゲット値と内部特徴量を使用する場合
        # encoder_vsn_weightの形状から推測（通常は4つ：target, target_lag, target_mean, target_std など）
        encoder_dynamic = [
            'past_target (POS販売冊数)',
            'target_lag1',
            'target_rolling_mean',
            'target_rolling_std'
        ]

    return {
        "static_cat": static_cat,
        "static_real": static_real,
        "decoder": decoder_dynamic,
        "encoder": encoder_dynamic
    }

def get_model_inputs(
    predictor: PyTorchPredictor,
    test_list: List[Dict]
) -> Dict[str, torch.Tensor]:
    """
    データスライスからモデル入力バッチを作成する
    """
    # Use predictor.predict() to generate forecasts, which internally handles transformation
    # However, we need to capture intermediate tensors, so we'll use a simpler approach

    # Create dataset and use predictor's internal data loading
    inference_dataset = ListDataset(test_list[:predictor.batch_size], freq="D")

    # Get transformation chain from predictor
    try:
        transformation = predictor.input_transform
    except AttributeError:
        transformation = predictor.transformation

    # Apply transformation to the dataset
    transformed_data = transformation.apply(inference_dataset, is_train=False)

    # Convert to list and get batch
    batch_data = list(transformed_data)[:predictor.batch_size]

    # Stack tensors manually to create batch
    device = predictor.device

    # Helper function to stack
    def stack_or_pad(key, data_list):
        values = [d[key] for d in data_list if key in d]
        if not values:
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values).to(device)
        else:
            return torch.tensor(np.stack(values)).to(device)

    # Stack all features
    inputs = {
        "feat_static_cat": stack_or_pad("feat_static_cat", batch_data),
        "feat_static_real": stack_or_pad("feat_static_real", batch_data),
        "feat_dynamic_real": stack_or_pad("feat_dynamic_real", batch_data),
        "past_target": stack_or_pad("past_target", batch_data),
        "past_observed_values": stack_or_pad("past_observed_values", batch_data),
    }

    # Add past_feat_dynamic_real if exists
    past_feat = stack_or_pad("past_feat_dynamic_real", batch_data)
    if past_feat is not None:
        inputs["past_feat_dynamic_real"] = past_feat

    # Remove None values
    inputs = {k: v for k, v in inputs.items() if v is not None}

    return inputs


def plot_variable_selection(
    weights: np.ndarray,
    feature_names: List[str],
    title: str,
    output_path: Path
):
    """
    Variable Selection Network (VSN) のアテンションをプロットする
    Shape can be:
    - 2D: [Batch, Num_Features]
    - 3D: [Batch, Time, Num_Features]
    """
    # バッチとタイムステップの全体で平均を取る
    if weights.ndim == 3:
        # [Batch, Time, Num_Features] -> [Num_Features]
        mean_weights = weights.mean(axis=(0, 1))
    elif weights.ndim == 2:
        # [Batch, Num_Features] -> [Num_Features]
        mean_weights = weights.mean(axis=0)
    else:
        print(f"Warning: Unexpected weight shape: {weights.shape}")
        return

    # Ensure feature_names length matches
    if len(feature_names) != len(mean_weights):
        print(f"Warning: Feature names ({len(feature_names)}) != weights ({len(mean_weights)})")
        feature_names = [f"Feature_{i}" for i in range(len(mean_weights))]

    df = pd.DataFrame({
        'Feature': feature_names,
        'Attention': mean_weights
    }).sort_values('Attention', ascending=False)

    plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
    sns.barplot(x='Attention', y='Feature', data=df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Attention Weight (Average over batch)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_temporal_self_attention(
    weights: np.ndarray,
    context_length: int,
    output_path: Path,
    sample_index: int = 0
):
    """
    Temporal Self-Attention のアテンション・マトリックスをプロットする
    Shape: [Batch, Query_Len, Key_Len]
    """
    # [Batch, Query_Len, Key_Len]
    attn_matrix = weights[sample_index, :, :]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        attn_matrix,
        cmap='hot',
        xticklabels=np.arange(1, attn_matrix.shape[1] + 1),
        yticklabels=np.arange(1, attn_matrix.shape[0] + 1),
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title(f'Temporal Self-Attention (Sample {sample_index})', fontsize=16)
    plt.xlabel('Key (Time Step)', fontsize=12)
    plt.ylabel('Query (Time Step)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


# ======================================================================
# メイン実行
# ======================================================================
def main():
    print("--- 1. Loading Model and Data ---")
    try:
        predictor = PyTorchPredictor.deserialize(Path("model"))
    except FileNotFoundError:
        print("ERROR: 'model/' directory not found. Run o_transformer.py first.")
        return

    try:
        dataset = pd.read_parquet('dataset.parquet')
        # encoders = pickle.load(open('encoders.pkl', 'rb')) # デコードには不要
    except FileNotFoundError:
        print("ERROR: 'dataset.parquet' not found. Run o_transformer.py (MAKE_DATASET=True) first.")
        return
        
    # 分析用の出力ディレクトリを作成
    output_dir = Path("attention_analysis")
    output_dir.mkdir(exist_ok=True)

    print("--- 3. Preparing Input Batch ---")
    # アテンション分析用に少数のスライスを（拡張なしで）生成
    test_slices = create_slices(
        dataset,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        region_extend=REGION_EXTEND,
        augmentation_factor=1 # 拡張なし
    )
    
    if not test_slices:
        print("ERROR: No data slices created. Check dataset.parquet.")
        return
        
    # モデル入力バッチを作成
    # （ここでは予測はしないが、フォワードパスのために必要）
    model_inputs = get_model_inputs(predictor, test_slices)

    print("--- 2. Preparing Feature Names ---")
    feature_names = get_feature_names(dataset, model_inputs)
    print(f"  Static: {len(feature_names['static_cat']) + len(feature_names['static_real'])} features")
    print(f"  Encoder: {len(feature_names['encoder'])} features")
    print(f"  Decoder: {len(feature_names['decoder'])} features")

    print("--- 4. Registering Forward Hooks ---")
    # PyTorchモデル本体を取得
    model = predictor.prediction_net.model
    model.eval() # 評価モードに設定

    hooks = []
    # (1) 時系列アテンション (MultiheadAttention)
    hooks.append(
        model.temporal_decoder.attention.register_forward_hook(
            get_attention_hook("self_attention")
        )
    )
    # (2) 静的変数アテンション - weight_networkから取得
    hooks.append(
        model.static_selector.weight_network.register_forward_hook(
            get_attention_hook("static_vsn_weight")
        )
    )
    # (3) 過去の動的変数アテンション - weight_networkから取得
    hooks.append(
        model.ctx_selector.weight_network.register_forward_hook(
            get_attention_hook("encoder_vsn_weight")
        )
    )
    # (4) 将来の動的変数アテンション - weight_networkから取得
    hooks.append(
        model.tgt_selector.weight_network.register_forward_hook(
            get_attention_hook("decoder_vsn_weight")
        )
    )

    print(f"Registered {len(hooks)} hooks.")

    print("--- 5. Running Forward Pass to Capture Attention ---")
    global attention_weights
    attention_weights = {} # キャプチャ変数をリセット
    
    with torch.no_grad():
        # フォワードパスを実行
        # これによりフックがトリガーされ、attention_weights に値が格納される
        _ = model(**model_inputs) 

    # フックを解除
    for h in hooks:
        h.remove()

    if not attention_weights:
        print("ERROR: Failed to capture attention. Model architecture may have changed.")
        return

    print("--- 6. Plotting Attention Weights ---")
    
    # (1) 時系列アテンションのプロット
    if "self_attention" in attention_weights:
        plot_temporal_self_attention(
            attention_weights["self_attention"],
            CONTEXT_LENGTH,
            output_dir / "temporal_self_attention_heatmap.png"
        )
        
    # (2) 静的変数アテンションのプロット
    # 注意: TFTの実装では static_cat と static_real は結合されてVSNに入力される
    if "static_vsn_weight" in attention_weights:
        static_features = feature_names["static_cat"] + feature_names["static_real"]
        plot_variable_selection(
            attention_weights["static_vsn_weight"],
            static_features,
            "Static Variable Selection (静的変数)",
            output_dir / "variable_selection_static.png"
        )

    # (3) 過去の動的変数アテンションのプロット (Encoder)
    if "encoder_vsn_weight" in attention_weights:
        plot_variable_selection(
            attention_weights["encoder_vsn_weight"],
            feature_names["encoder"], # 'POS販売冊数_roll_...'
            "Encoder Variable Selection (過去の動的変数)",
            output_dir / "variable_selection_encoder.png"
        )

    # (4) 将来の動的変数アテンションのプロット (Decoder)
    if "decoder_vsn_weight" in attention_weights:
        plot_variable_selection(
            attention_weights["decoder_vsn_weight"],
            feature_names["decoder"], # 'month_in_year_sin', 'mesh_pop_...'
            "Decoder Variable Selection (将来の動的変数)",
            output_dir / "variable_selection_decoder.png"
        )
        
    print(f"\nAnalysis complete. Results saved in: {output_dir}/")


if __name__ == "__main__":
    main()
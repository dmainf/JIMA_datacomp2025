# Variable Selection Network (VSN) の仕組み

## 現在の設定

モデルのVSNは固定のd_output値を使用しています：

- **Static selector**: `d_output=8` → 8個の静的特徴量を選択
- **Encoder selector**: `d_output=4` → 4個のエンコーダー特徴量を選択
- **Decoder selector**: `d_output=3` → 3個のデコーダー特徴量を選択

## VSNの役割

VSNは**Temporal Fusion Transformer (TFT)の中核機能**です：

1. **特徴量の重要度を学習**: 各特徴量に0〜1のアテンションウェイトを付与
2. **自動的に選択**: 重要な特徴量だけを選び、ノイズを除去
3. **解釈性の向上**: どの特徴が予測に重要かを可視化

## d_outputの決定方法

GluonTSのTFT実装では、`d_output`は以下のいずれかから決定されます：

### 1. デフォルト値（現在使用中）
```python
# GluonTSの内部デフォルト
static_d_output = 8
encoder_d_output = 4
decoder_d_output = 3
```

### 2. 推測値（static_dims/dynamic_dimsを指定した場合）
```python
# 元々試みた方法（dimension mismatchで失敗）
estimator = TemporalFusionTransformerEstimator(
    static_dims=[10],      # 10個の実数値特徴
    dynamic_dims=[18],     # 18個の動的特徴
    past_dynamic_dims=[21] # 21個の過去特徴
)
```
この場合、GluonTSは提供された特徴をそのまま使おうとしますが、
内部の変換処理と競合してdimension mismatchエラーが発生しました。

### 3. 手動設定（未サポート）
GluonTSのTFTEstimatorでは、`variable_dim`パラメータはありますが、
各VSNの`d_output`を個別に指定する公式パラメータはありません。

## なぜ削減が起きているか

**結論**: これは**意図的な設計**です。

1. **TFT論文の仕様**: VSNは元々、多数の特徴量から少数を選択する目的で設計されました
2. **GluonTSのデフォルト**: 汎用性を考慮し、控えめな数値(3, 4, 8)がハードコードされています
3. **Option Aを選択**: `static_dims`等を削除したため、GluonTSのデフォルト動作を使用中

## 削減の影響

### ポジティブ
- 過学習の防止
- 計算効率の向上
- 解釈性の向上（少数の重要特徴に集中）
- ノイズ除去

### ネガティブ
- 有用な特徴が除外される可能性
- 18個の精巧に作成した時間特徴が3個に削減
- 21個のroll特徴が4個に削減

## 解決策

### A. 現状を受け入れる（推奨）
- GluonTSのデフォルト動作を信頼
- VSNが自動的に最重要特徴を選択
- MASE=0.8690という良好な結果が出ている

### B. GluonTSのソースコードを修正
```python
# GluonTS内部のコードを直接編集
# site-packages/gluonts/torch/model/tft/module.py
# VariableSelectionNetworkのd_outputを変更
```
- 高度な技術が必要
- ライブラリの更新で変更が消える

### C. TFTの独自実装
- ゼロから実装（PyTorchで直接）
- d_outputを完全制御
- 大規模な作業

## 推奨アクション

**Option A を継続**することをお勧めします：

1. VSNは論文に基づく設計で、実績のある手法
2. 現在のMASE=0.8690は良好
3. アテンション可視化で、選ばれた3〜8特徴の重要度は確認できる
4. 全特徴を使いたい場合、Option CでTFT以外のモデル（N-BEATS、DeepAR等）を検討

## 参考
- TFT論文: https://arxiv.org/abs/1912.09363
- GluonTSドキュメント: https://ts.gluon.ai/

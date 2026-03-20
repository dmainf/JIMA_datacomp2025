import pandas as pd

df = pd.read_csv('../data/data.txt', sep='\t')

# normalize_authorで除外しているパターン
complex_patterns = [
    'さく・え', 'さく／え', '作・絵', '作／絵', '作絵',
    '著・画', '著・監修', '著・訳', '著・作詞', '著・作曲',
    '著・絵', '著／イラスト', '著＋訳', '著作',
    '企画・監修', '企画監修', '編・著', '編著'
]

single_patterns = [
    '監修', '原作', '漫画', 'イラスト',
    '著', '画', '作', '編', '訳', '文', '絵',
    'さく', 'マンガ', 'え', '〔著〕'
]

other_patterns = ['他著', '他']

print("=== 複合パターン ===")
for pattern in complex_patterns:
    count = df['著者名'].str.contains(pattern, na=False).sum()
    print(f"'{pattern}': {count}件")

print("\n=== 単独パターン（末尾） ===")
for pattern in single_patterns:
    count = df['著者名'].str.endswith(pattern, na=False).sum()
    print(f"'{pattern}': {count}件")

print("\n=== その他パターン ===")
for pattern in other_patterns:
    count = df['著者名'].str.contains(pattern, na=False).sum()
    print(f"'{pattern}': {count}件")

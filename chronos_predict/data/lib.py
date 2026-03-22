import numpy as np
import pandas as pd
import gc

def print_df(df):
    print("=== データの形状 ===")
    print(df.shape)
    print("\n=== 欠損値の数 ===")
    print(df.isnull().sum())
    print("\n=== 欠損値の割合 (%) ===")
    print((df.isnull().sum() / len(df) * 100).round(2))
    print("\n=== データ型 ===")
    print(df.dtypes)
    print()


def drop_unsure(df):
    print("dropping unsure data...")
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('ISBN', axis=1)
    df = df.dropna(subset=['出版社', '書名', '著者名', '本体価格'], how='all').copy()
    print("Complete!")
    return df


def drop_unstore(df, drop_stores):
    print("dropping unstore data...")
    df = df[~df['書店コード'].isin(drop_stores)].copy()
    print("Complete!")
    return df


def drop_negative_sales(df):
    print("dropping records with negative sales...")
    before_records = len(df)
    df = df[df['POS販売冊数'] >= 0].copy()
    after_records = len(df)
    removed_records = before_records - after_records
    print(f"Removed {removed_records:,} records with POS販売冊数 < 0")
    print(f"Remaining: {after_records:,} records")
    print("Complete!")
    return df


def normalize_titles(df, normalized_title_list):
    print("normalizing titles...")
    before_count = df['書名'].nunique()
    before_records = len(df)
    title_mapping = normalized_title_list[['書名', 'normalized_title']].copy()
    df = df.merge(title_mapping, left_on='書名', right_on='書名', how='inner')
    df['書名'] = df['normalized_title']
    df = df.drop('normalized_title', axis=1)
    after_count = df['書名'].nunique()
    after_records = len(df)
    removed_books = before_count - after_count
    removed_records = before_records - after_records
    print(f"Removed {removed_books:,} books ({removed_records:,} records) not in normalized_title_list")
    print(f"Remaining: {after_count:,} books ({after_records:,} records)")
    print("Complete!")
    return df


def remove_volume_number(df):
    print("removing volume information from titles...")
    df = df.copy()
    before_counts = df['書名'].str.count('_').value_counts().to_dict()
    if before_counts.get(2, 0) != len(df):
        raise ValueError(f"処理前に「_」が2つでないものがあります: {before_counts}")
    def process_volume(title):
        if pd.isna(title):
            return title
        return title.rsplit('_', 1)[0]
    df['書名'] = df['書名'].apply(process_volume)
    after_counts = df['書名'].str.count('_').value_counts().to_dict()
    if after_counts.get(1, 0) != len(df):
        raise ValueError(f"処理後に「_」が1つでないものがあります: {after_counts}")
    print("Convert 「作品_シリーズ_巻数」 to 「作品_シリーズ」")
    print("Complete!")
    return df


def filter_by_total_sales(df, min_sales):
    print(f"filtering books with total sales < {min_sales}...")
    total_sales = df.groupby('書名')['POS販売冊数'].sum()
    valid_books = total_sales[total_sales >= min_sales].index
    before_count = df['書名'].nunique()
    before_records = len(df)
    df = df[df['書名'].isin(valid_books)].copy()
    after_count = df['書名'].nunique()
    after_records = len(df)
    removed_books = before_count - after_count
    removed_records = before_records - after_records
    print(f"Removed {removed_books:,} books ({removed_records:,} records) with total sales < {min_sales}")
    print(f"Remaining: {after_count:,} books ({after_records:,} records)")
    print("Complete!")
    return df


def normalize_author(df):
    print("normalize author...")
    import re
    patterns = [
        '著', '監修', '画', '作', '編', '漫画', '他著', '編著', '原作', '文', '他',
        'さく', '絵', '作・絵', '他編', '作画', '撮影', 'ぶん', '訳', 'え',
        '他編著', '他監修', '編集', '写真', '解説', '監', '調査執筆', 'マンガ',
        '原案', '写真・文', '講師', '総監修', '文・写真', 'イラスト', '文・絵',
        '編訳', '監修・著', 'まんが', '絵と文', '料理', '訳注', 'さく・え',
        '編・著', '原著', '校注', '他作', '小説', '他画', '脚本', '責任編集',
        '編集主幹', '詩', '執筆', '病態監修', '原作・絵', '他訳', '医学監修',
        '責任監修', '協力', '作絵', '書', '監訳', '他絵', '編集代表', '著・監修',
        '語り', '詞', '構成', '他文', 'ぶん・え', '絵・文', '作・画', '訳・解説',
        '著作', '述', '他編集', '構成・絵', '出題', '編曲', '作／絵', 'ストーリー',
        '翻訳', '著・演奏', '著・絵', '写真と文', '全訳注', '他監', '指導',
        '文と絵', '聞き手', '他校注', '企画', '選', '著・写真', '作家',
        '英文解説', 'マンガ・文', '考案', 'レシピ', '特別講師', '原作／絵',
        '制作', 'ことば', '監修・写', '著・画', '構成・文', '他マンガ',
        '栄養監修', '訳編', '監修・文', '特別編集', '訳・監修', '編纂',
        '医療解説', 'デザイン', '写真・著', '再話', '俳句', '塗り絵', '他脚本',
        '他著作', '脚本・絵'
    ]
    """出現回数1回
    '著・訳', '著／イラスト', '著＋訳', '企画・監修',
    '企画監修', 'さく／え', '漢字監修', '植物監修', '編集・執筆',
    '地図監修', '英語監修', '日本語監修', '医師監修', '著・イラスト',
    '記事監修', '解説・監修', '監修執筆', '文・画', '図案', '手本・監修',
    '監修・考案', '俳句監修', '監修代表', '代表監修', '編集・文', '絵・著',
    '監修・協力', 'イラスト・文', '他全訳注', '他選・文', '漫画・文',
    '編修主幹', '写真・監修', '監修・執筆', '執筆・解説', '監修・解説',
    '編修代表', '編集協力', '取材・文', '撮影・監修', '監修・制作',
    '編集・構成', '編集責任', '監修・料理', '食事指導', '監修・作画',
    '絵・監訳', '作詞者', '家紋監修', 'しかけ', 'にんぎょう・え', '編修',
    '他料理', '他写真', '調理', 'パズル制作', '編訳・解説', '編・訳',
    '文・漫画', 'ぶ', '語り手', '他編集委員', '他編集代表',
    '手本', '著・作詞', '著・作曲', '他選', '編・写真', 'ノベライズ',
    '文・構成', '他講師', '編著代表', '他調査執筆', '編者代表', '主幹',
    '原訳', '校訂・訳', '主編', '編著・監修', '編・解説', '御作曲',
    '序', '原詩', '画と文', '原案＆作画', '原案・文', '他校訂・訳',
    'え・ぶん', '原案・解説', '監・著', '監修協力', '影絵', '校訂', '案',
    '他え', '訳・著', '技術指導', '朗読', '解説・訳', '注訳', 'しゃしん',
    '詩・文', '短歌・文', '講義', '詩人', '訳・注', '現代語訳', '著・装画',
    '監修訳注', '他執筆', '板書', '本文', '案と絵', '絵と案', '文章',
    '他原作', 'コミック', '選・著', '原作・原案', '原案・脚本', '監督',
    '推薦', '文と写真', '原作・脚本', '編／写真', '芝居', 'ほか著',
    '企画・編集', '改訂監修', '再話・絵', '指導・編曲', '他責任編集',
    '写真協力', '編注', '原著作', '主幹編著', '〔著〕'
    """
    patterns_sorted = sorted(patterns, key=len, reverse=True)
    combined_pattern = '|'.join(re.escape(p) for p in patterns_sorted)
    regex = re.compile(f'　(?:{combined_pattern})$')
    df['著者名'] = (df['著者名'].str.replace(regex, '', regex=True).str.replace('　', '', regex=False))
    print("Complete!")
    return df


def fill_unknown_author(df):
    print("filling missing author names with 'UNKNOWN'...")
    missing_count = df['著者名'].isna().sum()
    if df['著者名'].dtype.name == 'category':
        if 'UNKNOWN' not in df['著者名'].cat.categories:
            df['著者名'] = df['著者名'].cat.add_categories(['UNKNOWN'])
    df['著者名'] = df['著者名'].fillna('UNKNOWN')
    print(f"Filled {missing_count:,} missing author names with 'UNKNOWN'")
    print("Complete!")
    return df


def delete_space(df, delete_columns):
    df = df.copy()
    df[delete_columns] = df[delete_columns].apply(
        lambda x: x.str.replace(r'[ 　]+', '', regex=True)
    )
    return df


def convert_to_datetime(df):
    print("converting date column to datetime...")
    df['日付'] = pd.to_datetime(df['日付'].str[:10])
    print("Complete!")
    return df


def convert_to_category(df):
    print("converting object columns to category...")
    cols = df.select_dtypes(include=['object']).columns
    df[cols] = df[cols].astype('category')
    print("Complete!")
    return df


def calc_book_attrs(df, attrs):
    print("=== Calc book attributes ===")
    temp = df[['書名'] + attrs].copy()
    book_attrs = temp.groupby('書名', observed=False)[attrs].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
    book_attrs['書名_base'] = book_attrs['書名'].astype(str).str.split('_').str[0]
    del temp
    gc.collect()
    print("Complete!")
    return book_attrs


def calc_sales(df, keys):
    print("=== Calc sales ===")
    sales = df.groupby(keys, observed=False)['POS販売冊数'].sum().reset_index()
    gc.collect()
    print("Complete!")
    return sales


def make_full_index(book_attrs, start='2024-01-01', end='2024-12-31'):
    print("=== Make full index ===")
    dates = pd.date_range(start, end, freq='D')
    books = book_attrs['書名'].unique()
    print(f"Books: {len(books):,}")
    books_df = pd.DataFrame({'書名': books, 'key': 0})
    dates_df = pd.DataFrame({'日付': dates, 'key': 0})
    full_index = books_df.merge(dates_df, on='key').drop('key', axis=1)
    print(f"Full index size: {len(full_index):,}")
    print("Complete!")
    return full_index


def add_sales(full_index, sales, keys):
    print("=== Add sales ===")
    df = full_index.merge(sales, on=keys, how='left')
    df['POS販売冊数'] = df['POS販売冊数'].fillna(0).astype(np.int32)
    del sales
    gc.collect()
    print("Complete!")
    return df


def add_book_attrs(df, book_attrs):
    print("=== Add book attributes ===")
    df = df.merge(book_attrs, on='書名', how='left')
    print(f"Shape: {df.shape}")
    print("Complete!")
    return df


def add_time_features(df):
    print("\n=== Add time features ===")
    df['month'] = df['日付'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['dayofweek'] = df['日付'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
    df = df.drop(['month', 'dayofweek'], axis=1)
    print("Created features:")
    print("  - month_sin, month_cos")
    print("  - dayofweek_sin, dayofweek_cos")
    print("Complete!")
    return df

def convert_types(df):
    print("=== Convert types ===")
    int_cols = df.select_dtypes(include=['int64']).columns.tolist()
    float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    df[int_cols] = df[int_cols].astype(np.int32)
    df[float_cols] = df[float_cols].astype(np.float32)
    print("Complete!")
    return df

def clean_categories(df):
    print("=== Clean categories ===")
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in cat_cols:
        before = df[col].cat.categories.size
        df[col] = df[col].cat.remove_unused_categories()
        after = df[col].cat.categories.size
        print(f"{col}: {before} -> {after}")
    print("Complete!")
    return df

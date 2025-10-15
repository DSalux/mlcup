# -*- coding: utf-8 -*-
"""
Предобработка под мультимодальный пайплайн (CSV + текст + фото).
Создаёт:
  - data/processed/train_slim.parquet
  - data/processed/test_slim.parquet
  - отчёты по колонкам до/после (CSV)
Требует: pandas, numpy, pyarrow
Запуск:
  python prepare_tables.py --train data/raw/train.csv --test data/raw/test.csv --outdir data/processed
"""
import re
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# семейства признаков, у которых оставляем только 30-дневные окна
FAM = [
    "ExemplarAcceptedCountTotal",
    "ExemplarReturnedCountTotal",
    "ExemplarReturnedValueTotal",
    "GmvTotal",
    "OrderAcceptedCountTotal",
    "item_count_sales",
    "item_count_returns",
    "item_count_fake_returns",
]

ID_COLS: List[str] = ["id", "ItemID", "SellerID"]   # храним в таблице (ключи), но потом НЕ даём модели
TARGET = "resolution"

def slim(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    X = df.copy()

    # 1) текст: name_rus + description -> text
    if {"name_rus", "description"}.issubset(X.columns):
        X["text"] = (X["name_rus"].fillna("") + " " + X["description"].fillna("")).str.strip()

    # 2) убрать 7/90, оставить только 30-дневные окна
    drop = []
    for f in FAM:
        pat = re.compile(rf"^{re.escape(f)}(Total)?(7|30|90)$", re.I)
        cols = [c for c in X.columns if pat.match(c)]
        drop += [c for c in cols if not c.endswith("30")]
    X.drop(columns=[c for c in drop if c in X], inplace=True, errors="ignore")

    # 3) рейтинги -> агрегаты
    rcols = [c for c in X.columns if re.fullmatch(r"rating_[1-5]_count", c)]
    if rcols:
        R = X[rcols].fillna(0)
        w = np.array([int(c.split("_")[1]) for c in rcols])  # веса 1..5
        X["ratings_total"] = R.sum(axis=1)
        X["avg_rating"] = np.divide(
            R.mul(w, axis=1).sum(axis=1),
            X["ratings_total"].replace(0, np.nan)
        ).fillna(0)
        X.drop(columns=rcols, inplace=True)

    # 4) активности -> сумма (NaN -> 0)
    eng = [c for c in ["comments_published_count", "photos_published_count", "videos_published_count"] if c in X.columns]
    if eng:
        X[eng] = X[eng].fillna(0)
        X["engagement_total"] = X[eng].sum(axis=1)
        X.drop(columns=eng, inplace=True)

    # 5) категории -> строки + 'unknown'
    for c in ["brand_name", "CommercialTypeName4"]:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("unknown")

    # 6) даункост чисел (экономия памяти/диска)
    for c in X.select_dtypes(include=["int64", "int32"]).columns:
        X[c] = pd.to_numeric(X[c], downcast="integer")
    for c in X.select_dtypes(include=["float64", "float32"]).columns:
        X[c] = pd.to_numeric(X[c], downcast="float")

    # 7) порядок: ключи/таргет впереди (остаются в таблице, но НЕ идут в признаки модели)
    front = [c for c in ID_COLS if c in X.columns]
    if is_train and TARGET in X.columns:
        front += [TARGET]
    rest = [c for c in X.columns if c not in front]
    return X[front + rest]

def report(df: pd.DataFrame, path: Path) -> None:
    rep = (
        pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "n_unique": df.nunique(dropna=False),
        })
        .assign(null_frac=lambda t: (t.nulls / len(df)).round(6))
    )
    rep.to_csv(path, index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Путь к train.csv")
    ap.add_argument("--test",  required=True, help="Путь к test.csv")
    ap.add_argument("--outdir", default="data/processed", help="Куда сохранять parquet и отчёты")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # читаем
    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)

    # отчёты до
    report(tr, out / "columns_report_train_before.csv")
    report(te, out / "columns_report_test_before.csv")

    # slim
    tr_s = slim(tr, is_train=True)
    te_s = slim(te, is_train=False)

    # отчёты после
    report(tr_s, out / "columns_report_train_after.csv")
    report(te_s, out / "columns_report_test_after.csv")

    # сохраняем
    tr_s.to_parquet(out / "train_slim.parquet", index=False)
    te_s.to_parquet(out / "test_slim.parquet", index=False)

    print("Saved:", out / "train_slim.parquet", tr_s.shape, "|", out / "test_slim.parquet", te_s.shape)


if __name__ == "__main__":
    main()
    
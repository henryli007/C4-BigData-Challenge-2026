from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from advanced.features import DATE_COL, ID_COL, normalize_stock_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline semantic factors from THS hot stock text.")
    parser.add_argument("--hot-stock", default="./model/extra_apr_data/ths_hot_stock.csv")
    parser.add_argument("--output", default="./model/extra_apr_event_data/event_semantic_factors.csv")
    parser.add_argument("--report", default="")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    return parser.parse_args()


def _clean_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.replace(r"（免责声明：.*?）", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _load_text(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={ID_COL: str}, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    missing = [col for col in [DATE_COL, ID_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[ID_COL] = df[ID_COL].map(normalize_stock_code)
    reason = _clean_text(df["ths_hot_reason"]) if "ths_hot_reason" in df.columns else ""
    concept = _clean_text(df["ths_hot_concept"]) if "ths_hot_concept" in df.columns else ""
    text = (reason + " " + concept).str.strip()
    out = df[[DATE_COL, ID_COL]].copy()
    out["text"] = text
    out = out.drop_duplicates([DATE_COL, ID_COL], keep="last").reset_index(drop=True)
    return out


def _basic_text_stats(text: pd.Series) -> pd.DataFrame:
    length = text.str.len().astype(float)
    digit_count = text.str.count(r"\d").astype(float)
    sentence_count = text.str.count(r"[。；;！!？?]").astype(float)
    source_count = text.str.count(r"据|公告|年报|半年报|互动易|调研|纪要").astype(float)
    return pd.DataFrame(
        {
            "tsf_text_present": (length > 0).astype(float),
            "tsf_text_len_log": np.log1p(length),
            "tsf_text_digit_ratio": digit_count / length.clip(lower=1.0),
            "tsf_text_sentence_log": np.log1p(sentence_count),
            "tsf_text_source_log": np.log1p(source_count),
        },
        index=text.index,
    )


def main() -> None:
    args = parse_args()
    hot_path = Path(args.hot_stock)
    text_df = _load_text(hot_path)
    nonempty = text_df["text"].str.len() > 0
    corpus = text_df.loc[nonempty, "text"].tolist()
    if not corpus:
        raise ValueError(f"No non-empty text rows in {hot_path}")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )
    x = vectorizer.fit_transform(corpus)
    n_components = min(args.components, max(1, x.shape[1] - 1), max(1, x.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=2026)
    dense = svd.fit_transform(x)
    dense = StandardScaler().fit_transform(dense)

    sem = np.zeros((len(text_df), n_components), dtype=np.float32)
    sem[nonempty.to_numpy()] = dense.astype(np.float32)
    sem_cols = [f"tsf_text_svd_{i:02d}" for i in range(n_components)]
    sem_df = pd.DataFrame(sem, columns=sem_cols)
    stats = _basic_text_stats(text_df["text"])
    out = pd.concat([text_df[[DATE_COL, ID_COL]], stats, sem_df], axis=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    report = {
        "hot_stock": str(hot_path),
        "output": str(out_path),
        "rows": int(len(out)),
        "nonempty_rows": int(nonempty.sum()),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "components": int(n_components),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "date_min": str(out[DATE_COL].min().date()),
        "date_max": str(out[DATE_COL].max().date()),
    }
    report_path = Path(args.report) if args.report else out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

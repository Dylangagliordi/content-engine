"""Near line-for-line port of okn-analytics' scripts/models/ml_engine.py
(MLEngine) — same 14 ML models, same sklearn classes/hyperparameters, same
formulas. See okn_mirror/__init__.py for the "niche replaces platform"
rationale: this engine is instantiated ONCE PER NICHE (okn: once per platform).

Differences from upstream, every one explicit:
  - "platform" param/attribute is really a niche string here.
  - feature_cols drops okn's is_multilingual/has_emoji logic's Korean/Greek
    script-detection regex (qualume's content is English-only across all
    current niches) — has_emoji and caption_length/hashtag_count are kept
    verbatim since those are real, language-agnostic signals. is_multilingual
    kept as a column (always 0 for now) rather than removed, so the
    feature_cols list stays structurally comparable to upstream.
  - _caption_nlp drops the Greek/Korean stop-word blocks (stopwordsiso is
    still tried first if installed; falls back to sklearn's English list +
    the same social-media noise terms) — same multi-language-safe fallback
    structure, just without blocks for languages not present in this data.
  - saves is always 0 (see loader.py) — _anomaly_detection/_content_fatigue's
    quality_cols checks still run, just on an incomplete signal.
"""
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qualume.okn_mirror.config_okn import compute_recency_weights

logger = logging.getLogger("okn_mirror.ml")

# ══════════════════════════════════════════════
# SENTENCE EMBEDDINGS (optional — graceful fallback, same as upstream)
# ══════════════════════════════════════════════
_embedding_model = None
HAS_EMBEDDINGS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    pass


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None and HAS_EMBEDDINGS:
        logger.info("Loading multilingual embedding model...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model


def _compute_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        return model.encode(texts, show_progress_bar=False, batch_size=32)
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


class MLEngine:
    """Port of okn's MLEngine. Instantiate per-niche, same isolation
    discipline upstream applies per-platform (avoid methodology mixing
    across groups that aren't really comparable)."""

    MIN_POSTS_FOR_ML = 10
    MIN_POSTS_FOR_NN = 15

    def __init__(self, df: pd.DataFrame, platform: str):
        self.df = df.copy()
        self.platform = platform  # really a niche string, see module docstring
        self.results: Dict[str, Any] = {}
        self._extract_features()

    def _extract_features(self):
        df = self.df
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["hour"] = df["published_at"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["published_at"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        df["caption_length"] = df["title"].fillna("").str.len()
        df["hashtag_count"] = df["title"].fillna("").apply(lambda x: len(re.findall(r"#\w+", str(x))))
        df["has_emoji"] = df["title"].fillna("").apply(
            lambda x: 1 if re.search(r"[\U0001F600-\U0001F9FF\U00002600-\U000027BF\U0001F300-\U0001F5FF]", str(x)) else 0
        )
        # is_multilingual kept as a column for feature_cols structural parity
        # with upstream, but qualume's current niches are all English-only —
        # always 0, not a fabricated multilingual detector. See module docstring.
        df["is_multilingual"] = 0

        for ct in df["content_type"].unique():
            df[f"is_{ct}"] = (df["content_type"] == ct).astype(int)

        if "duration_sec" in df.columns:
            df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)

        self.df = df
        self.feature_cols = ["hour", "day_of_week", "is_weekend", "caption_length",
                              "hashtag_count", "has_emoji", "is_multilingual"]
        for col in df.columns:
            if col.startswith("is_") and col not in ["is_weekend", "is_multilingual"]:
                self.feature_cols.append(col)
        if "duration_sec" in df.columns:
            self.feature_cols.append("duration_sec")

        if "weight" in df.columns:
            self.sample_weights = df["weight"].values
        else:
            self.sample_weights = compute_recency_weights(df["published_at"]).values

        captions = df["title"].fillna("").tolist()
        self.embeddings = _compute_embeddings(captions)
        if self.embeddings is not None:
            logger.info("Embeddings: %sd vectors for %s captions", self.embeddings.shape[1], len(captions))
            from sklearn.decomposition import PCA
            # Real edge-case bug found via test_okn_mirror_engines.py, present
            # in upstream okn's identical formula too: n_components can exceed
            # min(n_samples, n_features) when len(captions) is very small
            # (e.g. 2), which PCA raises on rather than silently clamping.
            # Guarded here the same way every other okn stage degrades
            # gracefully (try/except -> status/skip) instead of crashing
            # __init__ before run_all's own MIN_POSTS_FOR_ML check ever runs.
            n_components = min(10, max(3, len(captions) // 5))
            max_valid = min(len(captions), self.embeddings.shape[1])
            if n_components > max_valid:
                logger.info(
                    "Skipping PCA-reduced embedding features: n_components=%s "
                    "exceeds max_valid=%s for %s captions", n_components, max_valid, len(captions),
                )
            else:
                pca = PCA(n_components=n_components, random_state=42)
                emb_reduced = pca.fit_transform(self.embeddings)
                for i in range(n_components):
                    col_name = f"emb_{i}"
                    df[col_name] = emb_reduced[:, i]
                    self.feature_cols.append(col_name)
                self.pca = pca

    def run_all(self) -> Dict[str, Any]:
        n = len(self.df)
        logger.info("ML Engine (%s): %s posts", self.platform, n)
        self.results["platform"] = self.platform
        self.results["n_posts"] = n

        if n < self.MIN_POSTS_FOR_ML:
            self.results["status"] = "insufficient_data"
            self.results["message"] = f"Need at least {self.MIN_POSTS_FOR_ML} posts for ML (have {n})"
            return self.results
        self.results["status"] = "ok"

        self.results["feature_importance"] = self._feature_importance()
        self.results["engagement_prediction"] = self._engagement_predictor()
        self.results["content_clusters"] = self._content_clustering()
        self.results["anomalies"] = self._anomaly_detection()
        self.results["caption_analysis"] = self._caption_nlp()
        self.results["engagement_drivers"] = self._engagement_drivers()
        self.results["content_fatigue"] = self._content_fatigue()
        self.results["posting_cadence"] = self._optimal_cadence()
        self.results["momentum_score"] = self._momentum_score()
        self.results["root_cause"] = self._root_cause_analysis()

        if self.embeddings is not None:
            self.results["topic_discovery"] = self._topic_discovery()
            self.results["similar_posts"] = self._similar_post_predictor()
            self.results["hashtag_clusters"] = self._hashtag_cluster_strategy()
        else:
            for key in ["topic_discovery", "similar_posts", "hashtag_clusters"]:
                self.results[key] = {"status": "no_embeddings"}
        return self.results

    # 1. ENGAGEMENT PREDICTOR (Neural Network)
    def _engagement_predictor(self) -> Dict:
        n = len(self.df)
        if n < self.MIN_POSTS_FOR_NN:
            return {"status": "need_more_data", "min_required": self.MIN_POSTS_FOR_NN}
        X = self.df[self.feature_cols].fillna(0).values
        y = self.df["engagement_rate"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation="relu", solver="adam",
                            max_iter=2000, random_state=42, early_stopping=True,
                            validation_fraction=0.15, alpha=0.01)
        try:
            if n >= 20:
                cv_scores = cross_val_score(mlp, X_scaled, y, cv=min(5, n // 4), scoring="r2")
                cv_r2 = float(np.mean(cv_scores))
            else:
                cv_r2 = None
            mlp.fit(X_scaled, y)
            y_pred = mlp.predict(X_scaled)
            r2 = float(r2_score(y, y_pred))
            mae = float(mean_absolute_error(y, y_pred))
            self.df["predicted_engagement_rate"] = y_pred
            self.df["engagement_residual"] = y - y_pred
            overperformers = self.df.nlargest(3, "engagement_residual")
            underperformers = self.df.nsmallest(3, "engagement_residual")
            return {
                "status": "trained", "model": "MLP Neural Network (32->16->8)",
                "r2_score": round(r2, 4),
                "cv_r2_score": round(cv_r2, 4) if cv_r2 is not None else None,
                "mae": round(mae, 4),
                "interpretation": self._interpret_r2(cv_r2 if cv_r2 is not None else r2),
                "overperformers": [
                    {"title": str(row.get("title", "") or "")[:80],
                     "actual": round(row["engagement_rate"], 4),
                     "predicted": round(row["predicted_engagement_rate"], 4),
                     "surplus": round(row["engagement_residual"], 4),
                     "permalink": row.get("permalink", "")}
                    for _, row in overperformers.iterrows()
                ],
                "underperformers": [
                    {"title": str(row.get("title", "") or "")[:80],
                     "actual": round(row["engagement_rate"], 4),
                     "predicted": round(row["predicted_engagement_rate"], 4),
                     "deficit": round(row["engagement_residual"], 4),
                     "permalink": row.get("permalink", "")}
                    for _, row in underperformers.iterrows()
                ],
            }
        except Exception as e:
            logger.warning("Neural net failed: %s", e)
            return {"status": "failed", "error": str(e)}

    # 2. FEATURE IMPORTANCE
    def _feature_importance(self) -> Dict:
        X = self.df[self.feature_cols].fillna(0).values
        y = self.df["engagement_rate"].values
        try:
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            gb.fit(X, y, sample_weight=self.sample_weights)
            importances = gb.feature_importances_
            feature_ranking = sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
            readable_names = {
                "hour": "Posting Hour", "day_of_week": "Day of Week", "is_weekend": "Weekend Post",
                "caption_length": "Caption Length", "hashtag_count": "Number of Hashtags",
                "has_emoji": "Uses Emoji", "is_multilingual": "Multilingual Caption",
                "duration_sec": "Video Duration", "is_organic": "Organic Content",
                "is_selling": "Selling Content", "is_unclassified": "Unclassified Content",
            }
            top_features = []
            for f, imp in feature_ranking[:8]:
                if imp <= 0.01:
                    continue
                entry = {"feature": readable_names.get(f, f), "raw_feature": f,
                         "importance": round(float(imp), 4), "pct": round(float(imp) * 100, 1)}
                # Real gap this closes (user: "whats the point of [embeddings]
                # if we cant see them"): raw emb_N is an opaque PCA-reduced
                # direction through a 384-dim sentence embedding — no human
                # can act on "emb_2 matters 55%". Attach a real interpretation
                # by contrasting the actual captions at each end of that axis.
                if f.startswith("emb_"):
                    component_idx = int(f.split("_")[1])
                    entry["interpretation"] = self._explain_embedding_component(component_idx)
                top_features.append(entry)
            return {
                "status": "ok", "model": "GradientBoosting (100 trees)",
                "r2_score": round(float(gb.score(X, y)), 4),
                "top_features": top_features,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _explain_embedding_component(self, component_idx: int, n_examples: int = 5, n_keywords: int = 5) -> Dict:
        """Make one PCA-reduced embedding dimension (emb_{component_idx})
        human-readable. The embedding vector itself has no inherent meaning
        (it's a learned direction through sentence-embedding space, not a
        named concept) — but the CAPTIONS at the extreme ends of that axis
        do. Standard embedding-interpretability technique: sort real
        captions by their projection onto this one PCA component, take the
        top/bottom-N, and extract what they have in common (TF-IDF keywords
        + the raw example captions themselves) so a human can read "high
        emb_2 = X-style captions, low emb_2 = Y-style captions" instead of
        an opaque number.

        Requires self.pca and self.embeddings (set in _extract_features when
        sentence-transformers succeeded and n_components didn't exceed
        max_valid — see that method's docstring). Returns a dict with
        high_keywords/low_keywords/high_examples/low_examples; never raises
        — any internal failure degrades to empty lists, same honest-gap
        discipline as the rest of this engine."""
        try:
            if not hasattr(self, "pca") or self.embeddings is None:
                return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}

            component_vector = self.pca.components_[component_idx]
            projections = self.embeddings @ component_vector

            order = np.argsort(projections)
            n = min(n_examples, len(order) // 2) if len(order) >= 2 else 0
            if n == 0:
                return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}

            high_idx = order[-n:][::-1]
            low_idx = order[:n]

            titles = self.df["title"].fillna("").tolist()
            high_captions = [titles[i] for i in high_idx]
            low_captions = [titles[i] for i in low_idx]

            def _top_keywords(captions: List[str]) -> List[str]:
                non_empty = [c for c in captions if len(c.strip()) > 2]
                if len(non_empty) < 2:
                    return []
                try:
                    tfidf = TfidfVectorizer(max_features=n_keywords, stop_words="english",
                                             token_pattern=r"(?u)\b[#\w][\w]{2,}\b")
                    tfidf.fit(non_empty)
                    return list(tfidf.get_feature_names_out())
                except Exception:
                    return []

            return {
                "high_keywords": _top_keywords(high_captions),
                "low_keywords": _top_keywords(low_captions),
                "high_examples": [str(c)[:80] for c in high_captions[:3]],
                "low_examples": [str(c)[:80] for c in low_captions[:3]],
            }
        except Exception as e:
            logger.warning("Failed to explain embedding component %s: %s", component_idx, e)
            return {"high_keywords": [], "low_keywords": [], "high_examples": [], "low_examples": []}

    # 3. CONTENT CLUSTERING (KMeans — the "niche-performance-tier" clustering,
    # distinct from topic_discovery's semantic clustering, #11 below)
    def _content_clustering(self) -> Dict:
        cluster_features = ["engagement_rate", "likes", "comments", "shares"]
        available = [c for c in cluster_features if c in self.df.columns]
        if len(available) < 2:
            return {"status": "insufficient_features"}
        X = self.df[available].fillna(0).values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        n = len(self.df)
        k = min(3, max(2, n // 5))
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.df["cluster"] = km.fit_predict(X_scaled)
            clusters = []
            for c in range(k):
                cluster_data = self.df[self.df["cluster"] == c]
                clusters.append({
                    "cluster_id": c, "size": len(cluster_data),
                    "avg_engagement_rate": round(cluster_data["engagement_rate"].mean(), 4),
                    "avg_reach": int(cluster_data["reach"].mean()),
                    "avg_likes": int(cluster_data["likes"].mean()),
                    "avg_shares": int(cluster_data["shares"].mean()),
                    "dominant_type": cluster_data["content_type"].mode().iloc[0] if len(cluster_data) > 0 else "unknown",
                    "sample_titles": cluster_data["title"].head(3).tolist(),
                })
            clusters.sort(key=lambda x: x["avg_engagement_rate"], reverse=True)
            labels = ["[TOP] Top Performers", "[AVG] Average", "[LOW] Needs Improvement"]
            for i, c in enumerate(clusters):
                c["label"] = labels[min(i, len(labels) - 1)]
            return {"status": "ok", "n_clusters": k, "clusters": clusters}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 4. ANOMALY DETECTION (IsolationForest)
    def _anomaly_detection(self) -> Dict:
        features = ["reach", "engagement_total", "likes", "comments", "shares"]
        available = [c for c in features if c in self.df.columns]
        if len(available) < 3:
            return {"status": "insufficient_features"}
        X = self.df[available].fillna(0).values
        try:
            iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
            self.df["anomaly_score"] = iso.fit_predict(X)
            self.df["anomaly_raw_score"] = iso.decision_function(X)
            anomalies = self.df[self.df["anomaly_score"] == -1].sort_values("anomaly_raw_score")
            results = []
            for _, row in anomalies.iterrows():
                is_positive = row["engagement_rate"] > self.df["engagement_rate"].median()
                results.append({
                    "title": str(row.get("title", "") or "")[:80],
                    "type": "viral_outlier" if is_positive else "underperformer_outlier",
                    "engagement_rate": round(row["engagement_rate"], 4), "reach": int(row["reach"]),
                    "anomaly_score": round(float(row["anomaly_raw_score"]), 4),
                    "permalink": row.get("permalink", ""),
                })
            return {
                "status": "ok", "model": "IsolationForest", "total_anomalies": len(anomalies),
                "viral_outliers": [r for r in results if r["type"] == "viral_outlier"],
                "underperformer_outliers": [r for r in results if r["type"] == "underperformer_outlier"],
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 5. CAPTION NLP (TF-IDF)
    def _caption_nlp(self) -> Dict:
        captions = self.df["title"].fillna("").tolist()
        valid_mask = self.df["title"].fillna("").str.len() > 10
        if valid_mask.sum() < 5:
            return {"status": "insufficient_captions"}
        valid_df = self.df[valid_mask]
        valid_captions = valid_df["title"].tolist()
        try:
            try:
                from stopwordsiso import stopwords as sw_iso
                stop_words = sw_iso(["en"])
            except ImportError:
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                stop_words = set(ENGLISH_STOP_WORDS)
            stop_words.update([
                "fyp", "foryou", "foryoupage", "viral", "trending", "reels",
                "like", "follow", "share", "comment", "link", "bio", "dm",
                "http", "https", "www", "com",
            ])
            tfidf = TfidfVectorizer(max_features=50, min_df=2, stop_words=list(stop_words),
                                     token_pattern=r"(?u)\b[#\w][\w]{2,}\b")
            tfidf_matrix = tfidf.fit_transform(valid_captions)
            feature_names = tfidf.get_feature_names_out()
            engagement = valid_df["engagement_rate"].values
            term_engagement = {}
            for i, term in enumerate(feature_names):
                term_presence = (tfidf_matrix[:, i].toarray().flatten() > 0).astype(int)
                if term_presence.sum() >= 2:
                    eng_with = engagement[term_presence == 1].mean()
                    eng_without = engagement[term_presence == 0].mean()
                    lift = (eng_with / eng_without) - 1 if eng_without > 0 else 0
                    term_engagement[term] = {
                        "term": term, "posts_with_term": int(term_presence.sum()),
                        "avg_engagement_with": round(float(eng_with), 4),
                        "avg_engagement_without": round(float(eng_without), 4),
                        "engagement_lift": round(float(lift), 4),
                    }
            high_engagement_terms = sorted(term_engagement.values(), key=lambda x: x["engagement_lift"], reverse=True)
            hashtags = [t for t in high_engagement_terms if t["term"].startswith("#")]
            return {
                "status": "ok", "top_engagement_terms": high_engagement_terms[:10],
                "top_hashtags": hashtags[:5],
                "low_engagement_terms": sorted(term_engagement.values(), key=lambda x: x["engagement_lift"])[:5],
                "total_terms_analyzed": len(term_engagement),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 6. ENGAGEMENT DRIVERS
    def _engagement_drivers(self) -> Dict:
        drivers = {}
        if len(self.df) >= 10:
            corr = self.df["caption_length"].corr(self.df["engagement_rate"])
            optimal_length = self.df.nlargest(min(5, len(self.df) // 3), "engagement_rate")["caption_length"].median()
            drivers["caption_length"] = {
                "correlation": round(float(corr), 4),
                "direction": "longer is better" if corr > 0.1 else "shorter is better" if corr < -0.1 else "no strong effect",
                "optimal_range": f"{int(optimal_length * 0.7)}-{int(optimal_length * 1.3)} characters",
            }
        if self.df["hashtag_count"].sum() > 0:
            corr = self.df["hashtag_count"].corr(self.df["engagement_rate"])
            optimal_tags = self.df.nlargest(min(5, len(self.df) // 3), "engagement_rate")["hashtag_count"].median()
            drivers["hashtags"] = {
                "correlation": round(float(corr), 4),
                "direction": "more is better" if corr > 0.1 else "fewer is better" if corr < -0.1 else "no strong effect",
                "optimal_count": int(optimal_tags),
            }
        if self.df["has_emoji"].sum() >= 2 and (self.df["has_emoji"] == 0).sum() >= 2:
            e_df = self.df[self.df["has_emoji"] == 1]
            ne_df = self.df[self.df["has_emoji"] == 0]
            emoji_eng = float(np.average(e_df["engagement_rate"].values, weights=e_df["weight"].values if "weight" in e_df.columns else None))
            no_emoji_eng = float(np.average(ne_df["engagement_rate"].values, weights=ne_df["weight"].values if "weight" in ne_df.columns else None))
            drivers["emoji"] = {
                "with_emoji_avg": round(float(emoji_eng), 4), "without_emoji_avg": round(float(no_emoji_eng), 4),
                "lift": round(float((emoji_eng / no_emoji_eng) - 1), 4) if no_emoji_eng > 0 else 0,
            }
        if self.df["is_weekend"].sum() >= 2 and (self.df["is_weekend"] == 0).sum() >= 2:
            we_df = self.df[self.df["is_weekend"] == 1]
            wd_df = self.df[self.df["is_weekend"] == 0]
            we_eng = float(np.average(we_df["engagement_rate"].values, weights=we_df["weight"].values if "weight" in we_df.columns else None))
            wd_eng = float(np.average(wd_df["engagement_rate"].values, weights=wd_df["weight"].values if "weight" in wd_df.columns else None))
            drivers["weekend_effect"] = {
                "weekend_avg": round(float(we_eng), 4), "weekday_avg": round(float(wd_eng), 4),
                "better": "weekend" if we_eng > wd_eng * 1.05 else "weekday" if wd_eng > we_eng * 1.05 else "no difference",
            }
        return drivers

    # 7. CONTENT FATIGUE
    def _content_fatigue(self) -> Dict:
        df = self.df.copy().sort_values("published_at")
        if len(df) < 15:
            return {"status": "need_more_data"}
        fatigue_results = []
        for ctype in df["content_type"].unique():
            cdf = df[df["content_type"] == ctype].copy()
            if len(cdf) < 5:
                continue
            cdf["days"] = (cdf["published_at"] - cdf["published_at"].min()).dt.total_seconds() / 86400
            eng = cdf["engagement_rate"].values
            days = cdf["days"].values
            w = cdf["weight"].values if "weight" in cdf.columns else np.ones(len(cdf))
            w_sum = w.sum()
            x_mean = np.average(days, weights=w)
            y_mean = np.average(eng, weights=w)
            numerator = np.sum(w * (days - x_mean) * (eng - y_mean))
            denominator = np.sum(w * (days - x_mean) ** 2)
            slope = 0 if denominator == 0 else numerator / denominator
            monthly_change = slope * 30
            if monthly_change < -0.01:
                status, severity = "declining", "high" if monthly_change < -0.03 else "medium"
            elif monthly_change > 0.01:
                status, severity = "growing", "positive"
            else:
                status, severity = "stable", "none"
            midpoint = len(cdf) // 2
            older_eng = cdf.iloc[:midpoint]["engagement_rate"].mean()
            recent_eng = cdf.iloc[midpoint:]["engagement_rate"].mean()
            fatigue_results.append({
                "content_type": ctype, "post_count": len(cdf), "trend": status, "severity": severity,
                "monthly_change_pct": round(float(monthly_change * 100), 2),
                "older_avg_rate": round(float(older_eng), 4), "recent_avg_rate": round(float(recent_eng), 4),
                "change_pct": round(float((recent_eng / older_eng - 1) * 100), 1) if older_eng > 0 else 0,
            })
        severity_order = {"high": 0, "medium": 1, "none": 2, "positive": 3}
        fatigue_results.sort(key=lambda x: severity_order.get(x["severity"], 2))
        return {
            "status": "ok", "content_types": fatigue_results,
            "fatigued_types": [r for r in fatigue_results if r["trend"] == "declining"],
            "growing_types": [r for r in fatigue_results if r["trend"] == "growing"],
        }

    # 8. OPTIMAL POSTING CADENCE
    def _optimal_cadence(self) -> Dict:
        df = self.df.copy().sort_values("published_at")
        if len(df) < 15:
            return {"status": "need_more_data"}
        df["week_start"] = df["published_at"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly = df.groupby("week_start").agg(
            post_count=("post_id", "count"), avg_engagement=("engagement_rate", "mean"),
            total_reach=("reach", "sum"), total_engagement=("engagement_total", "sum"),
        )
        if len(weekly) < 4:
            return {"status": "need_more_weeks"}
        cadence_perf = {}
        for count in sorted(weekly["post_count"].unique()):
            weeks_at_count = weekly[weekly["post_count"] == count]
            if len(weeks_at_count) >= 1:
                cadence_perf[int(count)] = {
                    "weeks_observed": len(weeks_at_count),
                    "avg_engagement_rate": round(float(weeks_at_count["avg_engagement"].mean()), 4),
                    "avg_total_reach": int(weeks_at_count["total_reach"].mean()),
                    "avg_total_engagement": int(weeks_at_count["total_engagement"].mean()),
                }
        reliable = {k: v for k, v in cadence_perf.items() if v["weeks_observed"] >= 2}
        pool = reliable if reliable else cadence_perf
        optimal = max(pool.items(), key=lambda x: x[1]["avg_engagement_rate"])
        max_reach = max(pool.items(), key=lambda x: x[1]["avg_total_reach"])
        current_cadence = round(weekly["post_count"].tail(4).mean(), 1)
        return {
            "status": "ok", "cadence_performance": cadence_perf,
            "optimal_for_engagement": {"posts_per_week": optimal[0], "avg_engagement_rate": optimal[1]["avg_engagement_rate"]},
            "optimal_for_reach": {"posts_per_week": max_reach[0], "avg_total_reach": max_reach[1]["avg_total_reach"]},
            "current_cadence": current_cadence,
            "recommendation": self._cadence_recommendation(current_cadence, optimal[0]),
        }

    @staticmethod
    def _cadence_recommendation(current, optimal):
        diff = optimal - current
        if abs(diff) < 0.5:
            return f"Current pace (~{current:.0f}/week) is close to optimal ({optimal}/week)."
        elif diff > 0:
            return f"Consider posting more: {optimal}/week is optimal, averaging {current:.0f}/week."
        else:
            return f"Consider posting less: {optimal}/week is optimal, averaging {current:.0f}/week."

    # 9. MOMENTUM SCORE
    def _momentum_score(self) -> Dict:
        df = self.df.copy().sort_values("published_at")
        if len(df) < 10:
            return {"status": "need_more_data"}
        now = df["published_at"].max()
        last_30 = df[df["published_at"] >= now - pd.Timedelta(days=30)]
        prev_30 = df[(df["published_at"] >= now - pd.Timedelta(days=60)) & (df["published_at"] < now - pd.Timedelta(days=30))]

        if len(last_30) >= 3 and len(prev_30) >= 3:
            eng_ratio = (last_30["engagement_rate"].mean() / prev_30["engagement_rate"].mean()) if prev_30["engagement_rate"].mean() > 0 else 1.0
            eng_score = min(25, max(0, (eng_ratio - 0.5) * 50))
        else:
            eng_score = 12.5

        weeks = df.groupby(df["published_at"].dt.to_period("W")).size()
        if len(weeks) >= 4:
            recent_weeks = weeks.tail(4)
            consistency = 1.0 - (recent_weeks.std() / recent_weeks.mean() if recent_weeks.mean() > 0 else 1.0)
            consistency_score = max(0, min(25, consistency * 25))
            if recent_weeks.min() > 0:
                consistency_score = min(25, consistency_score + 5)
        else:
            consistency_score = 10

        if len(last_30) >= 3 and len(prev_30) >= 3:
            reach_ratio = (last_30["reach"].mean() / prev_30["reach"].mean()) if prev_30["reach"].mean() > 0 else 1.0
            reach_score = min(25, max(0, (reach_ratio - 0.5) * 50))
        else:
            reach_score = 12.5

        quality_cols = [c for c in ["shares", "saves", "comments"] if c in df.columns]
        if quality_cols and len(last_30) >= 3 and len(prev_30) >= 3:
            recent_quality = last_30[quality_cols].sum().sum() / len(last_30)
            prev_quality = prev_30[quality_cols].sum().sum() / len(prev_30)
            quality_ratio = (recent_quality / prev_quality) if prev_quality > 0 else 1.0
            quality_score = min(25, max(0, (quality_ratio - 0.5) * 50))
        else:
            quality_score = 12.5

        total = round(eng_score + consistency_score + reach_score + quality_score, 1)
        if total >= 75:
            verdict = "Strong momentum"
        elif total >= 55:
            verdict = "Good momentum"
        elif total >= 35:
            verdict = "Moderate momentum"
        elif total >= 20:
            verdict = "Low momentum"
        else:
            verdict = "Warning — multiple signals declining"

        return {
            "status": "ok", "total_score": total, "max_possible": 100, "verdict": verdict,
            "breakdown": {
                "engagement_trend": round(eng_score, 1), "posting_consistency": round(consistency_score, 1),
                "reach_growth": round(reach_score, 1), "content_quality": round(quality_score, 1),
            },
            "detail": {
                "recent_30d_posts": len(last_30), "prev_30d_posts": len(prev_30),
                "recent_avg_engagement": round(float(last_30["engagement_rate"].mean()), 4) if len(last_30) > 0 else 0,
                "prev_avg_engagement": round(float(prev_30["engagement_rate"].mean()), 4) if len(prev_30) > 0 else 0,
            },
        }

    # 10. ROOT CAUSE ANALYSIS
    def _root_cause_analysis(self) -> Dict:
        if len(self.df) < self.MIN_POSTS_FOR_NN:
            return {"status": "need_more_data"}
        X = self.df[self.feature_cols].fillna(0).values
        y = self.df["engagement_rate"].values
        try:
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            gb.fit(X, y, sample_weight=self.sample_weights)
            predictions = gb.predict(X)
            residuals = y - predictions
            top_idx = np.argsort(residuals)[-3:][::-1]
            bottom_idx = np.argsort(residuals)[:3]
            feature_names = self.feature_cols
            readable_names = {
                "hour": "Posting Hour", "day_of_week": "Day of Week", "is_weekend": "Weekend Post",
                "caption_length": "Caption Length", "hashtag_count": "Hashtags", "has_emoji": "Emoji Usage",
                "is_multilingual": "Multilingual", "duration_sec": "Video Duration",
            }

            def _explain_post(idx):
                contributions = []
                for i, feat_name in enumerate(feature_names):
                    X_modified = X[idx].copy()
                    X_modified[i] = X[:, i].mean()
                    baseline_pred = gb.predict(X_modified.reshape(1, -1))[0]
                    actual_pred = predictions[idx]
                    contribution = actual_pred - baseline_pred
                    if abs(contribution) > 0.001:
                        readable = readable_names.get(feat_name, feat_name.replace("is_", "").replace("_", " ").title())
                        entry = {
                            "feature": readable, "value": float(X[idx][i]),
                            "contribution_pct": round(float(contribution * 100), 1),
                            "direction": "positive" if contribution > 0 else "negative",
                        }
                        # Same interpretability fix as #2's feature_importance:
                        # tell the reader WHAT this post's caption looked like
                        # relative to the axis, not just an opaque emb_N value.
                        if feat_name.startswith("emb_"):
                            component_idx = int(feat_name.split("_")[1])
                            axis = self._explain_embedding_component(component_idx)
                            side = "high" if entry["value"] >= 0 else "low"
                            entry["interpretation"] = {
                                "this_post_is_on_the": side,
                                "keywords_on_this_side": axis.get(f"{side}_keywords", []),
                            }
                        contributions.append(entry)
                contributions.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)
                return contributions[:5]

            viral_explanations = []
            for idx in top_idx:
                row = self.df.iloc[idx]
                viral_explanations.append({
                    "title": str(row["title"])[:60] if pd.notna(row["title"]) else "",
                    "actual_rate": round(float(y[idx]), 4), "predicted_rate": round(float(predictions[idx]), 4),
                    "surplus": round(float(residuals[idx]), 4), "permalink": row.get("permalink", ""),
                    "why": _explain_post(idx),
                })
            flop_explanations = []
            for idx in bottom_idx:
                row = self.df.iloc[idx]
                flop_explanations.append({
                    "title": str(row["title"])[:60] if pd.notna(row["title"]) else "",
                    "actual_rate": round(float(y[idx]), 4), "predicted_rate": round(float(predictions[idx]), 4),
                    "deficit": round(float(residuals[idx]), 4), "permalink": row.get("permalink", ""),
                    "why": _explain_post(idx),
                })
            return {"status": "ok", "viral_explanations": viral_explanations, "flop_explanations": flop_explanations}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 11. TOPIC DISCOVERY (semantic KMeans on caption embeddings)
    def _topic_discovery(self) -> Dict:
        if self.embeddings is None:
            return {"status": "no_embeddings"}
        df = self.df.copy()
        n = len(df)
        if n < 10:
            return {"status": "need_more_data"}
        try:
            n_clusters = min(max(3, n // 15), 8)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["topic_id"] = km.fit_predict(self.embeddings)
            topics = []
            for tid in range(n_clusters):
                mask = df["topic_id"] == tid
                topic_posts = df[mask]
                if len(topic_posts) == 0:
                    continue
                center = km.cluster_centers_[tid]
                topic_embeddings = self.embeddings[mask.values]
                dists = np.linalg.norm(topic_embeddings - center, axis=1)
                best_idx = np.argmin(dists)
                representative = str(topic_posts.iloc[best_idx].get("title", ""))[:80]
                captions = topic_posts["title"].fillna("").tolist()
                try:
                    tfidf = TfidfVectorizer(max_features=5, stop_words="english", token_pattern=r"(?u)\b[#\w][\w]{2,}\b")
                    tfidf.fit(captions)
                    keywords = list(tfidf.get_feature_names_out())
                except Exception:
                    keywords = []
                w = topic_posts["weight"].values if "weight" in topic_posts.columns else np.ones(len(topic_posts))
                w_sum = w.sum()
                avg_eng = float(np.average(topic_posts["engagement_rate"].values, weights=w)) if w_sum > 0 else 0
                topics.append({
                    "topic_id": int(tid), "post_count": len(topic_posts), "keywords": keywords,
                    "representative_post": representative, "avg_engagement_rate": round(avg_eng, 4),
                    "total_reach": int(topic_posts["reach"].sum()),
                    "content_types": topic_posts["content_type"].value_counts().to_dict(),
                })
            topics.sort(key=lambda x: x["avg_engagement_rate"], reverse=True)
            overall_avg = float(df["engagement_rate"].mean())
            return {
                "status": "ok", "n_topics": n_clusters, "topics": topics,
                "best_topic": topics[0] if topics else None, "worst_topic": topics[-1] if topics else None,
                "overall_avg_engagement": round(overall_avg, 4),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 12. SIMILAR POST PREDICTOR
    def _similar_post_predictor(self) -> Dict:
        if self.embeddings is None:
            return {"status": "no_embeddings"}
        df = self.df.copy()
        n = len(df)
        if n < 10:
            return {"status": "need_more_data"}
        try:
            sim_matrix = cosine_similarity(self.embeddings)
            np.fill_diagonal(sim_matrix, 0)
            predictions, errors = [], []
            for i in range(n):
                sim_scores = sim_matrix[i]
                top_k = min(5, n - 1)
                top_indices = np.argsort(sim_scores)[-top_k:][::-1]
                if len(top_indices) == 0:
                    continue
                neighbor_eng = df.iloc[top_indices]["engagement_rate"].values
                neighbor_sims = sim_scores[top_indices]
                predicted = float(np.average(neighbor_eng, weights=neighbor_sims)) if neighbor_sims.sum() > 0 else float(neighbor_eng.mean())
                actual = float(df.iloc[i]["engagement_rate"])
                error = actual - predicted
                errors.append(abs(error))
                predictions.append({"post_idx": i, "actual_rate": round(actual, 4), "predicted_rate": round(predicted, 4),
                                     "error": round(error, 4), "top_similarity": round(float(sim_scores[top_indices[0]]), 3)})
            predictions.sort(key=lambda x: abs(x["error"]))
            most_predictable = predictions[:3]
            most_surprising = predictions[-3:][::-1]
            mae = float(np.mean(errors)) if errors else 0
            median_sim = float(np.median(sim_matrix[sim_matrix > 0])) if (sim_matrix > 0).any() else 0
            recent_posts = []
            df_sorted = df.sort_values("published_at", ascending=False)
            for idx in df_sorted.index[:5]:
                i = df.index.get_loc(idx)
                sim_scores = sim_matrix[i]
                top_indices = np.argsort(sim_scores)[-3:][::-1]
                neighbor_eng = df.iloc[top_indices]["engagement_rate"].values
                neighbor_sims = sim_scores[top_indices]
                predicted = float(np.average(neighbor_eng, weights=neighbor_sims)) if neighbor_sims.sum() > 0 else float(neighbor_eng.mean())
                title = str(df.iloc[i].get("title", ""))[:60]
                similar_titles = [str(df.iloc[j].get("title", ""))[:40] for j in top_indices[:2]]
                recent_posts.append({
                    "title": title, "actual_rate": round(float(df.iloc[i]["engagement_rate"]), 4),
                    "predicted_rate": round(predicted, 4), "similar_to": similar_titles,
                    "similarity": round(float(sim_scores[top_indices[0]]), 3),
                })
            return {
                "status": "ok", "mean_absolute_error": round(mae, 4), "median_similarity": round(median_sim, 3),
                "most_predictable": most_predictable, "most_surprising": most_surprising,
                "recent_predictions": recent_posts,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # 13. HASHTAG CLUSTER STRATEGY
    def _hashtag_cluster_strategy(self) -> Dict:
        if self.embeddings is None:
            return {"status": "no_embeddings"}
        df = self.df.copy()
        all_hashtags = {}
        for idx, row in df.iterrows():
            caption = str(row.get("title", "") or "")
            tags = re.findall(r"#(\w{2,})", caption)
            eng_rate = float(row["engagement_rate"])
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower not in all_hashtags:
                    all_hashtags[tag_lower] = {"tag": f"#{tag_lower}", "posts": [], "engagement_rates": []}
                all_hashtags[tag_lower]["posts"].append(idx)
                all_hashtags[tag_lower]["engagement_rates"].append(eng_rate)
        if len(all_hashtags) < 3:
            return {"status": "insufficient_hashtags", "message": f"Only {len(all_hashtags)} unique hashtags found"}
        hashtag_texts = [v["tag"] for v in all_hashtags.values()]
        hashtag_keys = list(all_hashtags.keys())
        tag_embeddings = _compute_embeddings(hashtag_texts)
        if tag_embeddings is None:
            return {"status": "embedding_failed"}
        try:
            n_clusters = min(max(2, len(hashtag_keys) // 3), 6)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(tag_embeddings)
            clusters = []
            for cid in range(n_clusters):
                cluster_mask = labels == cid
                cluster_tags = [hashtag_keys[i] for i in range(len(hashtag_keys)) if cluster_mask[i]]
                if not cluster_tags:
                    continue
                all_eng, all_posts = [], set()
                for tag in cluster_tags:
                    info = all_hashtags[tag]
                    all_eng.extend(info["engagement_rates"])
                    all_posts.update(info["posts"])
                avg_eng = float(np.mean(all_eng)) if all_eng else 0
                sorted_tags = sorted(cluster_tags, key=lambda t: len(all_hashtags[t]["posts"]), reverse=True)
                clusters.append({
                    "cluster_id": int(cid), "top_hashtags": [f"#{t}" for t in sorted_tags[:5]],
                    "total_hashtags": len(cluster_tags), "total_posts": len(all_posts),
                    "avg_engagement_rate": round(avg_eng, 4), "label": " / ".join(sorted_tags[:3]),
                })
            clusters.sort(key=lambda x: x["avg_engagement_rate"], reverse=True)
            top_hashtags = []
            for tag, info in all_hashtags.items():
                if len(info["posts"]) >= 2:
                    top_hashtags.append({"tag": f"#{tag}", "post_count": len(info["posts"]),
                                          "avg_engagement": round(float(np.mean(info["engagement_rates"])), 4)})
            top_hashtags.sort(key=lambda x: x["avg_engagement"], reverse=True)
            return {
                "status": "ok", "n_clusters": n_clusters, "clusters": clusters,
                "best_cluster": clusters[0] if clusters else None,
                "top_individual_hashtags": top_hashtags[:10], "total_unique_hashtags": len(all_hashtags),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    @staticmethod
    def _interpret_r2(r2: float) -> str:
        if r2 >= 0.8:
            return "Excellent"
        elif r2 >= 0.5:
            return "Good"
        elif r2 >= 0.3:
            return "Moderate"
        elif r2 >= 0:
            return "Weak"
        else:
            return "Poor fit"


def run_ml(df: pd.DataFrame, platform: str) -> Dict[str, Any]:
    return MLEngine(df, platform).run_all()

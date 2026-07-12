# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.resources
import logging
import re
import joblib   # required to load XGBoost models saved with joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

from .rep_measure import rep_n_single
from .optimized_intra_with_sam import check_soft_repetition
from .harp_rep import Harp

class RepetitionModel:
    """Helper class for loading and running the ML-based repetition score model."""

    def __init__(self):
        """Load the XGBoost model, scaler, and feature info bundled inside the package
        using importlib.resources.
        """
        self.harp = Harp()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        except Exception as e:
            logger.error(f"Failed to load gpt2 tokenizer: {e}")
            self.tokenizer = None

        try:
            # reference the 'resources' directory within the repetition module
            resource_path = importlib.resources.files("omni_evaluator.evaluation.metrics.repetition").joinpath("resources")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Could not find the resources directory. Ensure the package is installed correctly."
            )

        # load feature_info.pkl
        with importlib.resources.as_file(resource_path.joinpath("feature_info.pkl")) as feature_info_path:
            if not feature_info_path.exists():
                raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")
            feature_info = joblib.load(feature_info_path)
            self.features = feature_info["features"]
            self.n_splits = feature_info["n_splits"]

        # load 5-fold models and scalers
        self.models = []
        self.scalers = []
        for fold in range(1, self.n_splits + 1):
            model_filename = f"xgboost_fold_{fold}.pkl"
            scaler_filename = f"scaler_fold_{fold}.pkl"
            
            with importlib.resources.as_file(resource_path.joinpath(model_filename)) as model_path:
                self.models.append(joblib.load(model_path))
                
            with importlib.resources.as_file(resource_path.joinpath(scaler_filename)) as scaler_path:
                self.scalers.append(joblib.load(scaler_path))

        logger.info("Repetition prob models loaded successfully.")

    def is_likely_markup(self, text: str) -> bool:
        """Heuristic to determine whether text is likely HTML or Markdown."""
        if text.count("<") > 10 and text.count(">") > 10:
            return True
        if text.count("\n## ") > 1 or text.count("\n* ") > 5:
            return True
        return False

    def _calculate_features(self, pred: str) -> dict:
        """Compute repetition indicator features for a single text."""
        text = str(pred)
        if self.is_likely_markup(text):
            text = re.sub(r"\s+", " ", text)

        if self.tokenizer:
            _, flagged_info = check_soft_repetition([text], tokenizer=self.tokenizer, threshold=0.0)
            intra_score, intra_freq_cnt = (
                (flagged_info[0]["score"], flagged_info[0]["count"]) if flagged_info else (0.0, 0)
            )
        else:
            intra_score, intra_freq_cnt = 0.0, 0

        harp_res = self.harp(text)
        hard_rep_char_score = len(harp_res) if harp_res else 0

        return {
            "rep-2": rep_n_single(text, 2),
            "rep-4": rep_n_single(text, 4),
            "intra_score": intra_score,
            "intra_freq_cnt": intra_freq_cnt,
            "hard_rep_char__score": hard_rep_char_score,
        }

    def predict_proba(self, pred: str) -> float:
        """Predict the repetition probability for a single text."""
        feature_dict = self._calculate_features(pred)
        test_df = pd.DataFrame([feature_dict], columns=self.features)
        X_test = test_df.values

        fold_probabilities = []
        for i in range(self.n_splits):
            X_test_scaled = self.scalers[i].transform(X_test)
            prob = self.models[i].predict_proba(X_test_scaled)[:, 1]
            fold_probabilities.append(prob)

        return float(np.mean(fold_probabilities))
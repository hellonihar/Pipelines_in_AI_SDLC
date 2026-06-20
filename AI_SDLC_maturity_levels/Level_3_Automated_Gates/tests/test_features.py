import numpy as np
import pytest
from src.features.build import build_features, get_preprocessor


class TestBuildFeatures:
    def test_preprocessor_output_shape(self, sample_df):
        preprocessor = get_preprocessor()
        X, y, _ = build_features(sample_df, preprocessor, fit=True)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] > 0

    def test_returns_y_as_array(self, sample_df):
        preprocessor = get_preprocessor()
        _, y, _ = build_features(sample_df, preprocessor, fit=True)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def test_fitted_transform_matches_fit_transform(self, sample_df):
        preprocessor = get_preprocessor()
        X_fit, _, fitted = build_features(sample_df, preprocessor, fit=True)

        X_transform, _, _ = build_features(sample_df, fitted, fit=False)
        np.testing.assert_array_equal(X_fit, X_transform)

    def test_unfitted_transform_raises(self, sample_df):
        preprocessor = get_preprocessor()
        with pytest.raises(Exception):
            build_features(sample_df, preprocessor, fit=False)

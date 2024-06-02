import unittest
import pandas as pd
from utils import get_best_params

class TestGetBestParams(unittest.TestCase):

    def setUp(self):
        """Sample data for testing get_best_params function."""
        data = {
            'mean_test_roc_auc':  [0.0, 0.0, 0.0, 0.0],
            'std_test_roc_auc':   [0.0, 0.0, 0.0, 0.0],
            'mean_test_f1':       [0.0, 0.0, 0.0, 0.0],
            'std_test_f1':        [0.0, 0.0, 0.0, 0.0],
            'mean_test_accuracy': [0.0, 0.0, 0.0, 0.0],
            'std_test_accuracy':  [0.0, 0.0, 0.0, 0.0],
            'param_degree':       [  1,   1,   1,   1]
        }
        self.parameters: list[str] = ['param_C', 'param_degree', 'param_coef0']
        self.cv_results: pd.DataFrame = pd.DataFrame(data)
        # self.cv_results[self.cv_results.columns.drop('param_degree')] = self.cv_results[self.cv_results.columns.drop('param_degree')].astype(np.float64)
        # self.cv_results['param_degree'] = self.cv_results['param_degree'].astype(int)

    def test_roc_auc_single_best_metric(self):
        """Test when there is a single best value for the metric"""
        self.cv_results.loc[1, 'mean_test_roc_auc'] = 1.0  # Set the row 1 to have the best value
        self.cv_results['param_degree'] = 2                # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1         # Set the degree of the best value to 1
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_roc_auc': 1.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_roc_auc_tie_in_metric_resolved_by_std(self):
        """Test when there is a tie resolved by the standard deviation of the metric"""
        self.cv_results['mean_test_roc_auc'] = 1.0         # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.03         # Set all std_test_roc_auc except the one that should be selected to 0.02
        self.cv_results.loc[2, 'std_test_roc_auc'] = 0.01  # Tiebreaker in std_test_roc_auc (row 2 should be selected)
        self.cv_results['param_degree'] = [0, 1, 2, 3]     #
        self.cv_results.loc[2, 'param_degree'] = 2         # This degree should be selected
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_roc_auc': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_roc_auc_tie_in_metric_and_std_resolved_by_fallback_metric(self):
        """Test when ties are resolved by the first fallback metric"""
        self.cv_results['mean_test_roc_auc'] = 1.0    # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1     # Tie in std_test_roc_auc
        self.cv_results.loc[3, 'mean_test_f1'] = 1.0  # Tiebreaker in mean_test_f1 (row 3 should be selected)
        self.cv_results['param_degree'] = 4           # Set all degrees except the one that should be selected to 4
        self.cv_results.loc[3, 'param_degree'] = 3    # This degree should be selected
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 3, 'mean_test_roc_auc': 1.0}, name=3)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_roc_auc_tie_in_metric_and_std_resolved_by_fallback_metric_std(self):
        """Test when ties are resolved by the first fallback metric std"""
        self.cv_results['mean_test_roc_auc'] = 1.0   # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1    # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 2.0         # Set all std_test_f1 except the one that should be selected to 0.0
        self.cv_results.loc[2, 'std_test_f1'] = 0.1  # Tiebreaker in std_test_f1 (row 4 should be selected)
        self.cv_results['param_degree'] = 1          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_roc_auc': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_roc_auc_tie_in_fallback_metric_std_resolved_by_second_fallback_metric(self):
        """Test when ties are resolved by the second fallback metric"""
        self.cv_results['mean_test_roc_auc'] = 1.0   # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1    # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_accuracy'] = 0.0  # Set all mean_test_accuracy except the one that should be selected to 0.0
        self.cv_results.loc[0, 'mean_test_accuracy'] = 1.0  # Tiebreaker in mean_test_accuracy (row 0 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[0, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_roc_auc': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_roc_auc_tie_in_fallback_metric_std_resolved_by_second_fallback_metric_std(self):
        """Test when ties are resolved by the second fallback metric std"""
        self.cv_results['mean_test_roc_auc'] = 1.0   # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1    # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Set all std_test_accuracy except the one that should be selected to 0.4
        self.cv_results.loc[1, 'std_test_accuracy'] = 0.2  # Tiebreaker in std_test_accuracy (row 1 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_roc_auc': 1.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_roc_auc_tie_resolved_by_param_degree(self):
        """Test when ties are resolved by the parameter degree"""
        self.cv_results['mean_test_roc_auc'] = 1.0   # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1    # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Tie in std_test_accuracy
        self.cv_results['param_degree'] = 3          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # Tiebreaker in param_degree (row 2 should be selected)
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_roc_auc': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_roc_auc_tie_resolved_by_selecting_first_element(self):
        """Test when ties are resolved by selecting the first element"""
        self.cv_results['mean_test_roc_auc'] = 1.0   # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1    # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Tie in std_test_accuracy
        self.cv_results['param_degree'] = 1          # Tie in param_degree
        best_params = get_best_params(self.cv_results, 'roc_auc', ['param_degree', 'mean_test_roc_auc'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_roc_auc': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_f1_single_best_metric(self):
        """Test when there is a single best value for the metric"""
        self.cv_results.loc[1, 'mean_test_f1'] = 1.0
        self.cv_results.loc[1, 'mean_test_f1'] = 2.0  # Set the row 1 to have the best value
        self.cv_results['param_degree'] = 2           # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1    # Set the degree of the best value to 1
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_f1': 2.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_f1_tie_in_metric_resolved_by_std(self):
        """Test when there is a tie resolved by the standard deviation of the metric"""
        self.cv_results['mean_test_f1'] = 1.0         # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.03         # Set all std_test_f1 except the one that should be selected to 0.02
        self.cv_results.loc[2, 'std_test_f1'] = 0.01  # Tiebreaker in std_test_roc_auc (row 2 should be selected)
        self.cv_results['param_degree'] = [0, 1, 2, 3]     #
        self.cv_results.loc[2, 'param_degree'] = 2         # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_f1': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_f1_tie_in_metric_and_std_resolved_by_fallback_metric(self):
        """Test when ties are resolved by the first fallback metric"""
        self.cv_results['mean_test_f1'] = 1.0    # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1     # Tie in std_test_f1
        self.cv_results.loc[3, 'mean_test_roc_auc'] = 1.0  # Tiebreaker in mean_test_roc_auc (row 3 should be selected)
        self.cv_results['param_degree'] = 4           # Set all degrees except the one that should be selected to 4
        self.cv_results.loc[3, 'param_degree'] = 3    # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 3, 'mean_test_f1': 1.0}, name=3)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_f1_tie_in_metric_and_std_resolved_by_fallback_metric_std(self):
        """Test when ties are resolved by the first fallback metric std"""
        self.cv_results['mean_test_f1'] = 1.0        # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 2.0         # Set all std_test_roc_auc except the one that should be selected to 0.0
        self.cv_results.loc[2, 'std_test_roc_auc'] = 0.1  # Tiebreaker in std_test_roc_auc (row 4 should be selected)
        self.cv_results['param_degree'] = 1          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_f1': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_f1_tie_in_fallback_metric_std_resolved_by_second_fallback_metric(self):
        """Test when ties are resolved by the second fallback metric"""
        self.cv_results['mean_test_f1'] = 1.0   # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_accuracy'] = 0.0  # Set all mean_test_accuracy except the one that should be selected to 0.0
        self.cv_results.loc[0, 'mean_test_accuracy'] = 1.0  # Tiebreaker in mean_test_accuracy (row 0 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[0, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_f1': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_f1_tie_in_fallback_metric_std_resolved_by_second_fallback_metric_std(self):
        """Test when ties are resolved by the second fallback metric std"""
        self.cv_results['mean_test_f1'] = 1.0   # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Set all std_test_accuracy except the one that should be selected to 0.4
        self.cv_results.loc[1, 'std_test_accuracy'] = 0.2  # Tiebreaker in std_test_accuracy (row 1 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_f1': 1.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_f1_tie_resolved_by_param_degree(self):
        """Test when ties are resolved by the parameter degree"""
        self.cv_results['mean_test_f1'] = 1.0   # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Tie in std_test_accuracy
        self.cv_results['param_degree'] = 3          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # Tiebreaker in param_degree (row 2 should be selected)
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_f1': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_f1_tie_resolved_by_selecting_first_element(self):
        """Test when ties are resolved by selecting the first element"""
        self.cv_results['mean_test_f1'] = 1.0   # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_accuracy'] = 0.0  # Tie in mean_test_accuracy
        self.cv_results['std_test_accuracy'] = 0.4   # Tie in std_test_accuracy
        self.cv_results['param_degree'] = 1          # Tie in param_degree
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_f1'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_f1': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_accuracy_single_best_metric(self):
        """Test when there is a single best value for the metric"""
        self.cv_results.loc[1, 'mean_test_accuracy'] = 1.0
        self.cv_results.loc[1, 'mean_test_accuracy'] = 2.0  # Set the row 1 to have the best value
        self.cv_results['param_degree'] = 2           # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1    # Set the degree of the best value to 1
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_accuracy': 2.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_accuracy_tie_in_metric_resolved_by_std(self):
        """Test when there is a tie resolved by the standard deviation of the metric"""
        self.cv_results['mean_test_accuracy'] = 1.0         # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.03         # Set all std_test_f1 except the one that should be selected to 0.02
        self.cv_results.loc[2, 'std_test_f1'] = 0.01  # Tiebreaker in std_test_roc_auc (row 2 should be selected)
        self.cv_results['param_degree'] = [0, 1, 2, 3]     #
        self.cv_results.loc[2, 'param_degree'] = 2         # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_accuracy': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_accuracy_tie_in_metric_and_std_resolved_by_fallback_metric(self):
        """Test when ties are resolved by the first fallback metric"""
        self.cv_results['mean_test_accuracy'] = 1.0    # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1     # Tie in std_test_f1
        self.cv_results.loc[3, 'mean_test_roc_auc'] = 1.0  # Tiebreaker in mean_test_roc_auc (row 3 should be selected)
        self.cv_results['param_degree'] = 4           # Set all degrees except the one that should be selected to 4
        self.cv_results.loc[3, 'param_degree'] = 3    # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 3, 'mean_test_accuracy': 1.0}, name=3)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_accuracy_tie_in_metric_and_std_resolved_by_fallback_metric_std(self):
        """Test when ties are resolved by the first fallback metric std"""
        self.cv_results['mean_test_accuracy'] = 1.0        # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1         # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 2.0         # Set all std_test_roc_auc except the one that should be selected to 0.0
        self.cv_results.loc[2, 'std_test_roc_auc'] = 0.1  # Tiebreaker in std_test_roc_auc (row 4 should be selected)
        self.cv_results['param_degree'] = 1          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_accuracy': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)

    def test_accuracy_tie_in_fallback_metric_std_resolved_by_second_fallback_metric(self):
        """Test when ties are resolved by the second fallback metric"""
        self.cv_results['mean_test_accuracy'] = 1.0   # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 0.0  # Set all mean_test_f1 except the one that should be selected to 0.0
        self.cv_results.loc[0, 'mean_test_f1'] = 1.0  # Tiebreaker in mean_test_f1 (row 0 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[0, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_accuracy': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_accuracy_tie_in_fallback_metric_std_resolved_by_second_fallback_metric_std(self):
        """Test when ties are resolved by the second fallback metric std"""
        self.cv_results['mean_test_accuracy'] = 1.0   # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 0.0  # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.4   # Set all std_test_f1 except the one that should be selected to 0.4
        self.cv_results.loc[1, 'std_test_f1'] = 0.2  # Tiebreaker in std_test_f1 (row 1 should be selected)
        self.cv_results['param_degree'] = 2          # Set all degrees except the one that should be selected to 2
        self.cv_results.loc[1, 'param_degree'] = 1   # This degree should be selected
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_accuracy': 1.0}, name=1)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_accuracy_tie_resolved_by_param_degree(self):
        """Test when ties are resolved by the parameter degree"""
        self.cv_results['mean_test_accuracy'] = 1.0   # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 0.0  # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.4   # Tie in std_test_f1
        self.cv_results['param_degree'] = 3          # Set all degrees except the one that should be selected to 3
        self.cv_results.loc[2, 'param_degree'] = 2   # Tiebreaker in param_degree (row 2 should be selected)
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 2, 'mean_test_accuracy': 1.0}, name=2)
        pd.testing.assert_series_equal(best_params, expected_params)


    def test_accuracy_tie_resolved_by_selecting_first_element(self):
        """Test when ties are resolved by selecting the first element"""
        self.cv_results['mean_test_accuracy'] = 1.0   # Tie in mean_test_accuracy
        self.cv_results['std_test_f1'] = 0.1    # Tie in std_test_f1
        self.cv_results['mean_test_roc_auc'] = 1.0        # Tie in mean_test_roc_auc
        self.cv_results['std_test_roc_auc'] = 0.1         # Tie in std_test_roc_auc
        self.cv_results['mean_test_f1'] = 0.0  # Tie in mean_test_f1
        self.cv_results['std_test_f1'] = 0.4   # Tie in std_test_f1
        self.cv_results['param_degree'] = 1          # Tie in param_degree
        best_params = get_best_params(self.cv_results, 'f1', ['param_degree', 'mean_test_accuracy'])
        expected_params = pd.Series({'param_degree': 1, 'mean_test_accuracy': 1.0}, name=0)
        pd.testing.assert_series_equal(best_params, expected_params)


if __name__ == '__main__':
    unittest.main()

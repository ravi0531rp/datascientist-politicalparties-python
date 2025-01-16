import pandas as pd
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names : list, n_components : int = 3):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.gmm_model = None
        self.n_components = n_components
        

    ##### YOUR CODE GOES HERE #####

    def fit_distribution(self):
        self.gmm_model = gaussianMixture(n_components = self.n_components)
        self.gmm_model.fit(self.data)

    def sample_distribution(self, num_samples = 10):
        return self.gmm_model.sample(num_samples)[0]

    def map_back(self, reduced_data):
        self.dim_reducer_model.inverse_transform(reduced_data)

    def predict(self, new_data):
        return np.exp(self.gmm_model.score_samples(new_data))

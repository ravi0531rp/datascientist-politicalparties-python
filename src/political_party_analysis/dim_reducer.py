import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd

class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.feature_columns = data.columns
        self.model = None

    ##### YOUR CODE GOES HERE #####
    def reduce_pca(self):
        self.model = PCA(n_components = self.n_components)
        reduced_data = self.model.fit_transform(self.data)
        return pd.DataFrame(reduced_data, columns = [f"PC_{i+1}" for i in range(self.n_components)])

    def inverse_transform(self, reduced_data):
        return pd.DataFrame(self.model.inverse_transform(reduced_data), columns = self.feature_columns)

    

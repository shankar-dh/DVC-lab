import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    return serialized_data
    

def data_preprocessing(data):

    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """
    df = pickle.loads(data)
    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return clustering_serialized_data


def build_save_model(data, filename):
    """
    Builds a KMeans clustering model, saves it to a file, and returns SSE values.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the clustering model.

    """
    df = pickle.loads(data)
    kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    max_val = float('-inf')
    true_clusters = 1
    for i in range(0, 49):
        if i - 1 >= 0 and i + 1 < 49:
            val = (sse[i] - sse[i + 1]) / (sse[i - 1] - sse[i])
            if val > max_val:
                max_val = val
                true_clusters = i
    kmeans = KMeans(n_clusters=true_clusters, **kmeans_kwargs)
    kmeans.fit(df)
    pickle.dump(kmeans, open(filename, 'wb'))



def load_model(filename):
    """
    Loads a saved KMeans clustering model.

    Args:
        filename (str): Name of the file containing the saved clustering model.


    Returns:
        str: A string indicating the predicted cluster.
    """

    loaded_model = pickle.load(open(filename, 'rb'))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    return "Cluster " + str(loaded_model.predict(df)[0])



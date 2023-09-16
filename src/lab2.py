import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle


def load_data():
    """

    :return:
    """

    # Read the CSV file into a DataFrame
    data = pd.read_csv("file.csv")

    return data


def data_preprocessing(data):
    data.isnull().sum()
    data = data.dropna()
    clustering_data = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    for i in clustering_data.columns:
        MinMaxScaler(i)
    return clustering_data


def build_save_model(data, filename):
    kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    pickle.dump(kmeans, open(filename, 'wb'))

    return sse

# def mapping_visual(data):
#     data["CREDIT_CARD_SEGMENTS"] = data["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1:
#         "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"})
#     PLOT = go.Figure()
#     for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
#         PLOT.add_trace(go.Scatter3d(x=data[data["CREDIT_CARD_SEGMENTS"] == i]['BALANCE'],
#                                     y=data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
#                                     z=data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],
#                                     mode='markers', marker_size=6, marker_line_width=1,
#                                     name=str(i)))
#     PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')
#
#     PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
#                        scene=dict(xaxis=dict(title='BALANCE', titlefont_color='black'),
#                                   yaxis=dict(title='PURCHASES', titlefont_color='black'),
#                                   zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')),
#                        font=dict(family="Gilroy", color='black', size=12))

def load_model(filename):
    """
    Loads a saved machine learning model from a file using pickle.

    Args:
        filename (str): The name of the file containing the saved model.

    Returns:
        object: The loaded machine learning model.
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def elbow(sse):
    kl = KneeLocator(
        range(1, 50), sse, curve="convex", direction="decreasing"
    )

    return "Number of clusters "+str(kl.elbow)



def main():
    data = load_data()
    process_data = data_preprocessing(data)
    sse = build_save_model(process_data, "model2.sav")
    model = load_model("model2.sav")
    test_data = pd.read_csv("test.csv")
    model.predict(test_data)
    print(elbow(sse))

    return model.predict(test_data)

    # X = scale_input(X)
    # X_train, y_train, X_test, y_test = data_preprocessing(X, y)
    # build_save_model("finalized_model.sav", X_train, y_train)
    # model = load_model("finalized_model.sav")
    # a = random.random()
    # b = random.random()
    # l = []
    # l.append(calculator.fun1(a, b))
    # l.append(calculator.fun2(a, b))
    # l.append(calculator.fun3(a, b))
    # l.append(calculator.fun4(l[0], l[1], l[2]))
    # l = np.array(l)
    # input = scale_input(l)
    # return process_data


if __name__ == "__main__":
    print(main())

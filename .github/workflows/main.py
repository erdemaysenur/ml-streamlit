import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
import plotly.express as px
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.svm import SVC

iris_wiki = "The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula 'all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus.\n\n The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other."
mnist_wiki = "The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by 're-mixing' the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels."
wine_desc = "These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines."
project = ""

st.set_page_config(layout="centered", page_icon="♥")

with st.sidebar:

    st.subheader("DATASET")
    dataset_option = st.selectbox(label="", options=("None", "Iris", "MNIST", "Wine"))

    st.subheader("MODEL")
    model_option = st.selectbox(label="",
                                options=("None", "Support Vector Machines", "Decision Tree", "Random Forests"))

if model_option == "None":
    if dataset_option == "None":
        st.header("Project")
        st.write(project)

    if dataset_option == "Iris":
        st.header("IRIS")
        st.subheader("Description")
        st.write(iris_wiki)
        data = load_iris(as_frame=True)
        X = data.data
        y = data.target
        num_features = X.shape[1]
        name_features = X.columns.to_list()
        st.subheader("Overview")
        st.dataframe(X.head())
        st.dataframe(X.describe())
        st.markdown("##### Targets:")
        st.write(', '.join(i for i in data.target_names))
        st.markdown("##### Feature counts:")
        st.write(num_features)
        st.markdown("##### Feature names:")
        st.write(', '.join(i for i in name_features))
        st.markdown("##### Null values:")
        st.write(X.isnull().sum())

        st.subheader("Exploring Dataset")
        df = px.data.iris()
        fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
                                color="species", template="ggplot2")
        st.plotly_chart(fig)

        for column in ["sepal_width", "sepal_length", "petal_width", "petal_length"]:
            fig = px.histogram(df, x=column, color="species", marginal="rug", hover_data=df.columns, template="ggplot2")
            st.plotly_chart(fig)

        fig = px.pie(names=y.value_counts().index, values=y.value_counts().to_numpy(), title='Target Distributions', template="ggplot2")
        st.plotly_chart(fig)

    if dataset_option == "MNIST":
        st.header("MNIST")
        st.subheader("Description")
        st.write(mnist_wiki)
        data = load_digits(n_class=10, as_frame=True)
        X = data.data
        y = data.target
        num_features = X.shape[1]
        images = data.images
        st.subheader("Overview")
        st.dataframe(X.describe())
        st.markdown("##### Targets:")
        st.write(', '.join(str(i) for i in data.target_names))


        st.subheader("Exploring Dataset")
        fig = px.pie(names=y.value_counts().index, values=y.value_counts().to_numpy(), title='Target Distributions', template="ggplot2")
        st.plotly_chart(fig)

        fig = plt.figure(figsize=(3,3))
        subplot = 331
        for i in range(9):
            plt.gray()
            fig.add_subplot(subplot)
            plt.imshow(images[i])
            plt.axis("off")
            plt.title(str(i), fontsize=3)
            subplot += 1
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle("Digit Samples", fontsize=4)
        st.pyplot(fig)
        density = X.describe().loc["mean"].values.reshape(8, 8)
        fig, ax = plt.subplots()
        plt.title("Pixel densities")
        ax = sns.heatmap(density, cmap="gray")
        st.pyplot(fig)

    if dataset_option == "Wine":
        st.header("Wine Dataset")
        st.subheader("Description")
        st.write(wine_desc)
        st.markdown("<a href='https://archive.ics.uci.edu/ml/datasets/wine'>Source</a>", unsafe_allow_html=True)
        data = load_wine(as_frame=True)
        X = data.data
        y = data.target
        num_features = X.shape[1]
        name_features = X.columns.to_list()
        st.subheader("Overview")
        st.dataframe(X.head())
        st.dataframe(X.describe())
        st.markdown("##### Targets:")
        st.write(', '.join(i for i in data.target_names))
        st.markdown("##### Feature counts:")
        st.write(num_features)
        st.markdown("##### Feature names:")
        st.write(', '.join(i for i in name_features))
        st.markdown("##### Null values:")
        st.write(X.isnull().sum())

        st.subheader("Exploring Dataset")
        df = pd.concat([X,y], axis=1)
        for column in X.columns.values:
            fig = px.histogram(df, x=column, color="target", marginal="rug", hover_data=df.columns, template="ggplot2")
            st.plotly_chart(fig)

        fig = px.pie(names=y.value_counts().index, values=y.value_counts().to_numpy(), title='Target Distributions', template="ggplot2")
        st.plotly_chart(fig)

elif model_option == "Support Vector Machines":
    st.header("Models")
    st.write("ldfkgtjıefokwpldğş")
    st.header("Train")
    st.write(f"Your choice of dataset: {dataset_option}")
    train = st.button("Train")
    if train:
        model = SVC()
        model.fit(X.values, y.values)



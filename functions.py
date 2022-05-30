import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris_wiki = "The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula 'all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus.\n\n The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other."
mnist_wiki = "The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by 're-mixing' the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels."
wine_desc = "These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines."


def load_dataset(dataset_option):
    if dataset_option == "Iris":
        st.write(iris_wiki)
        data = load_iris(as_frame=True)
        X = data.data
        y = data.target
    elif dataset_option == "MNIST":
        st.write(mnist_wiki)
        data = load_digits(n_class=10, as_frame=True)
        X = data.data
        y = data.target
    elif dataset_option == "Wine":
        st.write(wine_desc)
        data = load_wine(as_frame=True)
        X = data.data
        y = data.target
    return data, X, y

def load_page(dataset_option):
    if dataset_option == "Iris":
        st.write(iris_wiki)
        st.markdown("<a href='https://en.wikipedia.org/wiki/Iris_flower_data_set'>Source</a>", unsafe_allow_html=True)
    elif dataset_option == "MNIST":
        st.write(mnist_wiki)
        st.markdown("<a href='https://en.wikipedia.org/wiki/MNIST_database'>Source</a>", unsafe_allow_html=True)
    elif dataset_option == "Wine":
        st.write(wine_desc)
        st.markdown("<a href='https://archive.ics.uci.edu/ml/datasets/wine'>Source</a>", unsafe_allow_html=True)

def load_model_page(dataset_option, model_option, model_desc):
    st.header(model_option)
    st.write("Model description here")
    if dataset_option != "None":
        st.header("Train")
        st.write(f"Your choice of dataset: {dataset_option}")
        data, X, y = load_dataset(dataset_option)
        train = st.button("Start training")
        if train:
            model = SVC()
            model.fit(X.values, y.values)
            train_preds = model.predict(X.values)
            st.subheader("Report")
            cr = pd.DataFrame(classification_report(y.values, train_preds, output_dict=True))
            st.dataframe(cr)
            cm = confusion_matrix(y.values, train_preds)
            fig, ax = plt.subplots()
            plt.title("Confusion Matrix", fontsize=14)
            ax = sns.heatmap(cm, annot=True, fmt=".2f")
            st.pyplot(fig)


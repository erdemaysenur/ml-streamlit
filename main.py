import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
from functions import load_dataset, load_page, load_model_page
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

about = "This streamlit app was created to help machine learning beginners and coders who would like to contribute.\nMy motivation is to give a hint to a curios friend of mine about machine learning in a basic and sharable way. For now, there are limited dataset and model options, if you would like to increase these, any contributions are welcome!"
svc_desc = "In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Developed at AT&T Bell Laboratories by Vladimir Vapnik with colleagues (Boser et al., 1992, Guyon et al., 1993, Cortes and Vapnik, 1995, Vapnik et al., 1997) SVMs are one of the most robust prediction methods, being based on statistical learning frameworks or VC theory proposed by Vapnik (1982, 1995) and Chervonenkis (1974). Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces."
dt_desc = "A decision tree is a flowchart-like structure in which each internal node represents a 'test' on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules. In decision analysis, a decision tree and the closely related influence diagram are used as a visual and analytical decision support tool, where the expected values (or expected utility) of competing alternatives are calculated."
rf_desc = "Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance."

st.set_page_config(layout="centered", page_icon="♥")

with st.sidebar:

    st.subheader("DATASET")
    dataset_option = st.selectbox(label="", options=("None", "Iris", "MNIST", "Wine"))

    st.subheader("MODEL")
    model_option = st.selectbox(label="",
                                options=("None", "Support Vector Machine", "Decision Tree", "Random Forest"))

if model_option == "None":
    if dataset_option == "None":
        st.header("About")
        st.write(about)
        st.markdown("<a href='https://github.com/erdemaysenur/ml-streamlit'>Repo</a>", unsafe_allow_html=True)
        st.header("Datasets")
        st.write("There are 3 different dataset for now and to load them I used sklearn.datasets class."+"\n"+"Available datasets:"+"\n"+"- Iris"+"\n"+"- MNIST"+"\n"+"- Wine"+"\n"+"For more information select dataset in the sidebar you'd like to know about.")
        st.header("Models")
        st.write("There are 3 model imported from sklearn to train the dataset you chose."+"\n"+"- Support Vector Machines"+"\n"+"- Decision Tree"+"\n"+"- Random Forests"+"\n"+"After choosing dataset and model from sidebar, click 'Start Training' button and when model trained, classification report and confusion matrix will be shown below.")
    elif dataset_option == "Iris":
        st.header("IRIS")
        st.subheader("Description")
        data, X, y = load_dataset(dataset_option)
        load_page(dataset_option)
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

    elif dataset_option == "MNIST":
        st.header("MNIST")
        st.subheader("Description")
        data, X, y = load_dataset(dataset_option)
        load_page(dataset_option)
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

    elif dataset_option == "Wine":
        st.header("Wine Dataset")
        st.subheader("Description")
        data, X, y = load_dataset(dataset_option)
        load_page(dataset_option)
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

elif model_option == "Support Vector Machine":
    st.header(model_option)
    model = SVC()
    load_model_page(dataset_option, model, svc_desc)

elif model_option == "Decision Tree":
    st.header(model_option)
    model = DecisionTreeClassifier()
    load_model_page(dataset_option, model, dt_desc)

elif model_option == "Random Forest":
    st.header(model_option)
    model = RandomForestClassifier()
    load_model_page(dataset_option, model, rf_desc)





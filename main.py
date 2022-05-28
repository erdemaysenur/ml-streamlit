import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
from functions import load_dataset, load_page
import plotly.express as px
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

about = "This streamlit app was created to help machine learning beginners and coders who would like to contribute.\nMy motivation is to give a hint to a curios friend of mine about machine learning in a basic and sharable way. For now, there are limited dataset and model options, if you would like to increase these, any contributions are welcome!"

st.set_page_config(layout="centered", page_icon="♥")

with st.sidebar:

    st.subheader("DATASET")
    dataset_option = st.selectbox(label="", options=("None", "Iris", "MNIST", "Wine"))

    st.subheader("MODEL")
    model_option = st.selectbox(label="",
                                options=("None", "Support Vector Machines", "Decision Tree", "Random Forests"))

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

elif model_option == "Support Vector Machines":
    st.header("Support Vactor Machines")
    st.write("ldfkgtjıefokwpldğş")
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





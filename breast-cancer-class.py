import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("Breast Cancer Binary Classification")
    st.sidebar.title("Binary Classification")
    st.markdown("Predicting Breast Cancer with Naive Bayes Classifier 🍈")
    st.sidebar.markdown("Predicting Breast Cancer 🍈")
    run_classification()

def run_classification():
    # Load model breast cancer data
    data = load_breast_cancer()
    X = data['data']
    y = data['target']

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Precision-Recall curve", "ROC Curve", "Confusion Matrix"))

    unique, counts = np.unique(y, return_counts=True)
    st.write(f"Class Distribution: {dict(zip(unique, counts))}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Fit a Naive Bayes model
    clf = GaussianNB().fit(X_train, y_train)

    # Define fig variable
    fig = None

    if classifier == "Precision-Recall curve":
        # Predict probability for training set
        y_prob_train = clf.predict_proba(X_train)[:, 1]
        
        # Plot Precision-Recall curve
        fig, ax = plt.subplots()
        precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
        ax.fill_between(recall, precision)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.set_title("Train Precision-Recall curve")
        st.pyplot(fig)

    elif classifier == "ROC Curve":
        # Predict probability for test set
        y_prob_test = clf.predict_proba(X_test)[:, 1]

        # Plot ROC curve
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        auc_score = roc_auc_score(y_test, y_prob_test)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC={auc_score:.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='red')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

    elif classifier == "Confusion Matrix":
        # Predict probability for test set
        y_pred_test = clf.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        st.subheader("Confusion Matrix")
        st.write(cm)
        # Assuming `cm` is your confusion matrix
        cm_display = ConfusionMatrixDisplay(cm).plot()
        # Display the confusion matrix plot using st.pyplot()
        st.pyplot(cm_display.figure_)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Brest Cancer Data Set (Classification)")
        st.write(data)
        st.markdown("This [data set](https://goo.gl/U2Uwz2) includes Number of Attributes: 30 numeric, predictive attributes and the class.")
    
if __name__ == '__main__':
    main()

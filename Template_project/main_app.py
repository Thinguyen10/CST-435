import streamlit as st
import pandas as pd
from my_utils.data_loader import load_csv
from my_utils.cleaner import clean_df
from my_utils.train_test_split import prepare_features_target, split_data
from my_utils.model_evaluation import evaluate_model
from my_utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
)
from my_utils.hyperparameter_tuning import run_hyperparameter_search
import matplotlib.pyplot as plt

st.title("Universal ML Project App")

# === File Upload ===
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("No file uploaded. Using default dataset.")
    df = load_csv()  # from my_utils

st.write("### Raw Data Preview", df.head())

# === Data Cleaning ===
df = clean_df(df)

# === Feature + Target Selection ===
st.sidebar.header("Feature/Target Selection")
all_columns = df.columns.tolist()
target_column = st.sidebar.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)

# Encode target if needed
if df[target_column].dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

X, y = prepare_features_target(df, target_column=target_column)
feature_names = df.drop(columns=[target_column]).columns.tolist()

# === Train/Test Split ===
X_train, X_test, y_train, y_test = split_data(X, y)

# === Hyperparameters ===
st.sidebar.header("Hyperparameters")
# Generalized sliders/options for most ML models
max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000, step=100)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 1.0])

# === Model Selection ===
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose a Model", ["Perceptron", "Logistic Regression", "Decision Tree", "Random Forest", "KMeans"])

# === Train Button ===
if st.button("Train Model"):
    # Instantiate model based on selection
    if model_choice == "Perceptron":
        from my_utils.perceptron import create_perceptron, train_model
        model = create_perceptron(max_iter=max_iter, eta0=learning_rate)
        model = train_model(model, X_train, y_train)
    elif model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
    elif model_choice == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
    elif model_choice == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
    elif model_choice == "KMeans":
        from sklearn.cluster import KMeans
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters)
        model.fit(X)  # KMeans ignores y
        st.write("Cluster centers:", model.cluster_centers_)

    # === Evaluate (skip for unsupervised models like KMeans) ===
    if model_choice != "KMeans":
        acc, report, y_pred, metrics = evaluate_model(model, X_test, y_test)
        st.write(f"### Accuracy: {acc:.2f}")
        st.write("### Precision:", metrics["precision"])
        st.write("### Recall:", metrics["recall"])
        st.write("### F1 Score:", metrics["f1"])
        st.write("### Classification Report", report)

        # Visualization
        st.pyplot(plot_confusion_matrix(y_test, y_pred))
        st.pyplot(plot_roc_curve(model, X_test, y_test))
        st.pyplot(plot_precision_recall_curve(model, X_test, y_test))

        fig_importance = plot_feature_importance(model, feature_names)
        if fig_importance:
            st.pyplot(fig_importance)

# === Hyperparameter Search Button ===
if st.button("Run Hyperparameter Search") and model_choice != "KMeans":
    best_params, best_score = run_hyperparameter_search(X_train, y_train)
    st.write("### Hyperparameter Search Results")
    st.write("Best Params:", best_params)
    st.write("Best Score:", best_score)

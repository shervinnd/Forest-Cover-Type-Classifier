# 🌟 Forest Cover Type Prediction with Neural Networks 🌲

Welcome to the **Forest Cover Type Classifier** repository! This project
uses a TensorFlow-based neural network to predict forest cover types
from the UCI Covertype dataset. 🌳🔍 Whether you're into machine
learning, environmental data, or just love forests, this notebook has
you covered! We preprocess data, train a multi-layer perceptron,
evaluate performance, test random samples, and plot ROC curves for
insightful analysis. 🚀

## 📋 Project Overview

-   **Dataset**: UCI Forest Covertype (581,012 samples, 54 features, 7
    classes like Spruce/Fir, Lodgepole Pine, etc.) 🌿
-   **Model**: Sequential DNN with Dense layers, Dropout for
    regularization, and Softmax output. Trained with Adam optimizer and
    categorical cross-entropy. 🧠
-   **Key Features**:
    -   Data splitting & standardization 📊
    -   Batch training with TensorFlow Datasets ⚡
    -   Accuracy evaluation (\~85% on test set) ✅
    -   Random sample prediction testing 🎲
    -   Multi-class ROC curve visualization 📈
-   **Tech Stack**: TensorFlow, Scikit-learn, Matplotlib, NumPy 🛠️

## 🛠️ Installation

1.  Clone the repo:

    ``` bash
    git clone https://github.com/shervinnd/forest-cover-type-classifier.git
    cd forest-cover-type-classifier
    ```

2.  Install dependencies (use a virtual environment like venv or conda):

    ``` bash
    pip install tensorflow numpy matplotlib scikit-learn
    ```

3.  Open the Jupyter Notebook:

    ``` bash
    jupyter notebook covtype.ipynb
    ```

    *Note: This was tested on Python 3.12 with GPU acceleration (T4).
    Ensure TensorFlow is GPU-enabled if needed! ⚙️*

## 🚀 Usage

1.  **Run the Notebook**: Execute cells sequentially to:
    -   Import libraries 📚
    -   Load & preprocess data (fetch_covtype, scaling, one-hot
        encoding) 🔄
    -   Build & compile the model 🏗️
    -   Train for 20 epochs (batch size 128) ⏱️
    -   Evaluate on test set 📉
    -   Test a random sample 🎯
    -   Generate ROC curves for each class 📊
2.  **Customize**:
    -   Tweak hyperparameters like epochs, batch size, or layers in the
        parameters cell. 🔧
    -   Run `test_random_sample()` multiple times for fun predictions!
        😄
3.  **Output Example**:
    -   Training logs show accuracy improving to \~81% on train, \~85%
        on validation.
    -   ROC AUCs: High for most classes (e.g., 0.99+ for some)! 🌟

## 📊 Results & Insights

-   **Test Accuracy**: \~85.23% 🎉
-   **Sample Prediction**: Picks a random test instance, predicts cover
    type (e.g., Lodgepole Pine), shows probabilities, and checks
    correctness. ✅/❌
-   **ROC Curves**: Visualizes model confidence per class -- great for
    multi-class imbalance analysis! 📈 (Plotted with Matplotlib)
-   Pro Tip: Classes like Cottonwood/Willow might have lower AUC due to
    fewer samples. Experiment with oversampling! ⚖️

## 🤝 Contributing

We'd love your input! 🌍

-   Fork the repo & create a pull request.
-   Suggestions: Add more models (e.g., CNNs, XGBoost), hyperparameter
    tuning with Keras Tuner, or deployment with Streamlit. 💡
-   Report issues or bugs via GitHub Issues. 🐛

## 📄 License

This project is licensed under the MIT License -- feel free to use,
modify, and share! 📜

**Powered by Miracle ⚡** -- Exploring forests one prediction at a time!
🌲

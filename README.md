# Twitter Sentiment Analysis using Logistic Regression

## Project Overview
This project performs **sentiment analysis** on tweets using **Logistic Regression**. The model classifies tweets as **Positive ** or **Negative ** based on their textual content. The dataset consists of tweets with labeled sentiments, and the model is trained using **TF-IDF vectorization** combined with additional extracted features.

##Features
- Preprocesses tweets using **TF-IDF vectorization**
- Implements **Logistic Regression** for classification
- Extracts additional features (tweet length, punctuation count, etc.)
- Allows real-time sentiment analysis of user-input tweets
- Saves and loads the trained model using **Pickle**

##Technologies Used
- **Python**
- **scikit-learn** (Logistic Regression, TF-IDF)
- **NumPy** (Feature Engineering)
- **Pandas** (Data Handling)
- **Pickle** (Model Persistence)

## DataSet From
- **Kaggle- Sentiment140 dataset with 1.6 million tweets**

## Project Structure
```
📁 Twitter-Sentiment-Analysis
│── 📄 sentiment_analysis.py  # Main script for training and testing
│── 📄 requirements.txt        # Dependencies
│── 📄 preprocess.py           # Text preprocessing functions
│── 📄 train.py                # Model training and saving
│── 📄 predict.py              # Loads model and predicts sentiment
│── 📄 README.md               # Project documentation
│── 📂 data
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Testing dataset
│── 📂 models
│   ├── model.pkl             # Saved Logistic Regression model
│   ├── vectorizer.pkl        # Saved TF-IDF Vectorizer
```

## 🔧 Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model**
   ```bash
   python train.py
   ```
4. **Test user input tweets**
   ```bash
   python predict.py
   ```

## 🏆 Usage
### **Train the Model**
```python
python train.py
```
### **Predict Sentiment of a User Input Tweet**
```python
python predict.py
```
When prompted, enter a tweet, and the model will classify it as **Positive ** or **Negative **.

## 📊 Results & Model Performance
The trained model achieves an **accuracy of ~77%** on the test dataset. Performance can be improved by:
- Using more sophisticated **feature engineering**
- Experimenting with **different ML models**
- Training on a **larger dataset**

## 🤝 Contribution
Feel free to contribute to this project! Fork the repository, make changes, and submit a pull request.

---
🔗 **Author:** [Tanishtha_Reddy](https://github.com/Tanishtha-Reddy)


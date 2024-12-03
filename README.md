# **Course Recommendation System**

## **Overview**
This project is a hybrid course recommendation system designed to provide personalized and efficient recommendations for e-learning platforms. The system integrates advanced techniques, including sentiment analysis, clustering, collaborative filtering, and content-based filtering, to tackle the challenges of information overload and personalization in online education.

---

## **Project Structure**
| File/Folder                | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `requirements.txt`         | List of Python libraries required to run the project.                     |
| `DataAnalysis.ipynb`       | Notebook for data exploration, cleaning, and preprocessing.                |
| `SentimentAnalysis.ipynb`  | Notebook for sentiment analysis implementation using VADER.                |
| `Clustering.ipynb`         | Notebook for clustering and evaluating course clusters using K-Means.      |
| `DeepLearningApproach.ipynb` | Notebook for exploring deep learning models for sentiment analysis. |
| `recengine.ipynb`          | Notebook for implementing the recommendation system.                |
| `README.md`                | Documentation for the project.                                             |

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/Sciotaku/DTUDataScienceProject.git
```

### **2. Set Up a Virtual Environment**
```bash
python -m venv myenv
source myenv/bin/activate    # On Windows: myenv\Scripts\activate
```

### **3. Install Dependencies**
```bash
Install the required Python libraries using the requirements.txt file:
pip install -r requirements.txt
```

### **3. Install Dependencies**
Download Dataset from https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera

## **How to Run the Project**
The project involves three main steps executed sequentially in Jupyter notebooks:

### **1. Data Preprocessing**
Open and run the DataAnalysis.ipynb notebook.
This notebook loads the dataset, cleans and preprocesses the data, and prepares features for analysis.
The output will include a processed dataset ready for sentiment analysis.

### **2. Sentiment Analysis**
Open and run the SentimentAnalysis.ipynb notebook.
This notebook performs sentiment analysis using the VADER tool.
The sentiment scores are aggregated at the course level and stored for further use in the recommendation engine.

### **3. Recommendation Engine**
Open and run the recengine.ipynb notebook.
This notebook implements the hybrid recommendation system by combining clustering, and content-based filtering.
Provide an input course name, and the system will generate personalized course recommendations.

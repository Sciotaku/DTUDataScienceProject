# **Course Recommendation System**

## **Overview**
This project is a hybrid course recommendation system designed to provide personalized and efficient recommendations for e-learning platforms. The system integrates advanced techniques, including sentiment analysis, clustering, collaborative filtering, and content-based filtering, to tackle the challenges of information overload and personalization in online education.

### **Key Features**
- Sentiment analysis using VADER to analyze course reviews and prioritize highly rated courses.
- Clustering with K-Means to group similar courses for contextually relevant recommendations.
- Locality-Sensitive Hashing (LSH) for efficient similarity computations at scale.
- Dynamic filtering to allow users to customize recommendations based on ratings, institutions, or other preferences.
- Scalable and efficient processing for large datasets with over 1.5 million course reviews.

---

## **Project Structure**
| File/Folder                | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `myenv/`                   | Virtual environment setup for dependency isolation.                        |
| `requirements.txt`         | List of Python libraries required to run the project.                     |
| `DataAnalysis_Rahul.ipynb` | Notebook for data exploration, cleaning, and preprocessing.                |
| `SentimentAnalysis_Rahul.ipynb` | Notebook for sentiment analysis implementation using VADER.              |
| `Clustering_Rahul.ipynb`   | Notebook for clustering and evaluating course clusters using K-Means.      |
| `recengine.ipynb`          | Notebook for implementing the hybrid recommendation system.                |
| `recengine.py`             | Python script for the recommendation engine logic.                         |
| `sentiment_analysis.py`    | Script for running sentiment analysis on course reviews.                   |
| `README.md`                | Documentation for the project.                                             |
| `jobscript.sh`             | Script for running the project on a server or high-performance computing setup. |
| `job_error.log`            | Log file for errors during server execution.                               |
| `job_output.log`           | Log file for standard output during server execution.                      |

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
Open and run the DataAnalysis_Rahul.ipynb notebook.
This notebook loads the dataset, cleans and preprocesses the data, and prepares features for analysis.
The output will include a processed dataset ready for sentiment analysis.

### **2. Sentiment Analysis**
Open and run the SentimentAnalysis_Rahul.ipynb notebook.
This notebook performs sentiment analysis using the VADER tool.
The sentiment scores are aggregated at the course level and stored for further use in the recommendation engine.

### **3. Hybrid Recommendation Engine**
Open and run the recengine.ipynb notebook.
This notebook implements the hybrid recommendation system by combining clustering, LSH-based content similarity, and collaborative filtering.
Provide an input course name, and the system will generate personalized course recommendations.
#Importing project libraries
import pandas as pd
import matplotlib.pyplot as plt #this librariy for printing charts
import numpy as np
from sklearn.cluster import KMeans #this librariy is used for clustering and points

#this librariy for machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


def print_dataest():
     #this function is used for printing datase
     #Printing the initial dataset
    print("\n       --------------- Dataset --------------")
    print(df)
    print("\n")

def op_missing():
    #this function perform missing value handing operation
    # Cleaning the dataset by filling the missing values
    df.fillna({'Profession': 'None', 'Age': df['Age'].mean(), 'Score': df['Score'].mean()}, inplace=True)
    print("--- Printing clean dataset ---")
    print("\n")
    print(df) 


def EDA_data():
        # this function find out the maximum and minimum score in dataset (EDA)
        print("\n-- Maximum Scores in dataset : ", df['Score'].max())
        print("-- Minimum Scores in dataset : ", df['Score'].min())


def bar_graph():
        #printing the bar graph of the student-Professio
        plt.bar(df['Name'],df['Score'],color="orange",edgecolor="black")
        plt.xlabel("Name")
        plt.ylabel("Score")
        plt.title("Student-Profession")
        plt.show()


def cluster_scatter():
    #Clustering operation performed on the Score column
    model = KMeans(n_clusters=3, random_state=0, n_init=10)

    #Convert the 1D column into a 2D array using double brackets [['Score']]
    model.fit(df[['Score']])
    labels = model.labels_

    print("\n--- Printing cluster and points ---")
    for i in range(len(df['Score'])):
        #correctly pull the score value out of the DataFrame
        print(f"Score: {df['Score'].iloc[i]:.2f} --> Cluster {labels[i] + 1}")
    print("\n--- Printing the Cluster Centers ---")
    print("\n",model.cluster_centers_)
    #data is 1D, we plot Age on X-axis and Score on Y-axis to visualize the clusters
    x_co = df['Age']
    y_co = df['Score']

    # Creating the scatter plot (color-coded by cluster labels)
    plt.scatter(x_co, y_co, c=labels,edgecolors='black', cmap='viridis', s=100, zorder=2)

    # Optional: Add a line connecting the points chronologically if you want a line chart
    plt.plot(x_co,y_co,color="blue",linestyle="--",alpha=0.5,zorder=1)
    plt.title("Clustering (Score)")
    plt.xlabel("Age")
    plt.ylabel("Score")
    plt.grid(True,linestyle=":",alpha=0.6)
    plt.show()
    #printing the scatter graph that indecate score with repective age of the employ

def simple_linear_regression():
    #linear regression operation on data
    print("\n")
    print("----Performing Simple linear Regression Opeartion ----")
    print("\n")
    x=[]
    y=[]
    # # print(type(df['Score'])---> this is ''' pandas.Series'''

    for ag in df['Age']:
        x.append(ag)
    for sc in df['Score']:
        y.append(sc)

    # this is '''list'''
    # print(x)
    # print(y)

    #performing mathematical simple linear regression operation
    sum_x=sum(x)
    n=len(x)
    sum_y=sum(y)
    sum_xy=sum(x[i]*y[i] for i in range(n))
    sum_x2=sum(x[i]**2 for i in range(n))
    m=(n * sum_xy - sum_x * sum_y)/(n*sum_x2 - sum_x**2)
    c=(sum_y - m * sum_x)/n
    print("Slope is :- ",m)
    print("Intercept :- ",c)
    age=21
    prediction=m*age+c
    print("Prediction of score that depend on age :- ",prediction)
    if prediction >=3.8:
        print("Quality of the work is good ! Positive")
    else:
        print("Quality of work is bad ! Negative")

    #sentiment analysis based on the dadatset ''' if the score is greater thab 4.0 is positive on otherhand , less than 4.0 score is negative
def sentiment():
        #sentiment analysis 
    sentiment=[]
    for score in df["Score"]:
        if score > 4.0:
            sentiment.append("Positive")
        else:
            sentiment.append("Negative")
    df['Sentiment'] = sentiment

    #priting the dataset
    print("\n --[ Exiting Dataset Sentiment ]--")
    print(df[['Name','Score','Sentiment']])

    #---naive bayes that basedc on dataset sentiment
    x=df[['Score']].values
    y=df[['Sentiment']].values

    model=GaussianNB()
    model.fit(x,y)

    # testing on new dataset
    new_dataset=np.array([
        [4.5],
        [5.0],
        [3.7],
        [4.8],
        [3.6],
        [3.8]
    ])

    prediction=model.predict(new_dataset)

    #printing sentiment of new dataset

    print("\n--- Analyse the sentiment ---")
    print("\n-------------------------------")
    for score,predict in zip(new_dataset.flatten(),prediction):
        print(f"Score :---> {score:.1f} Sentiment : ----->{predict}")

#Initialising the dataset
data = pd.DataFrame({
    'Profession': ['Web Development','Application Development', 'Java Full Stack Developement', 'Python Developer', 'Python Full Stack Developement', 'Bug Bounty', 'Penetration Testing','Software Engineering','Software Developement','Database Manager','None'],
    'Name': ['Ayush.Varma', 'Krishna.Kumar', 'Arjun.Nagar', 'Priya.Sharma', 'Suresh.Rao', 'Vikram.Nagar','Ayush.Kumar','Ajinkay.Sharma','Vijay.Kumar','Ajay.Muria','Vijay.Nagar'],
    'Age': [22, 23, 24, 25, 27, np.nan, 28,27,26,28,np.nan],
    'Score': [4.1,4.8,4.2,4.5,4.2,4.1,3.2,2.0, np.nan,3.2,np.nan]
})

df = pd.DataFrame(data)

#classing functions
print_dataest()
op_missing()
EDA_data()
bar_graph()
simple_linear_regression()
cluster_scatter()
sentiment()

#basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# sklearn imports
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#our code
#import linear_model
import utils
import clean


def main():

    X = pd.read_csv('../data/BlackFriday.csv')# names =("User_ID", "Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status,", "Product_Category_1","Product_Category_2","Product_Category_3", "Purchase" ))
    N, d = X.shape
    X.info()
    X.sort_values('User_ID').head(10)
    X['User_ID'].value_counts().count()   #5,891 customers
    # fill missing values with 0
    # (?) need to calculate percentage of missing value?
    X = X.fillna(0)
    # change gender to 0 and 1
    X['Gender'] = X['Gender'].apply(change_gender)
    # change age to 0 to 6
    X['Age'] = X['Age'].apply(change_age)
    # change city categories to 0 to 2
    X['City_Category'] = X['City_Category'].apply(change_city)
    # change the year to integer
    X['Stay_In_Current_City_Years'] = X['Stay_In_Current_City_Years'].apply(change_year)
    
    
    
     #predict age
    # Make y matrix to be the age
    y = np.zeros((N,1))
    y = X.values[:,3]
    y = y.astype('int')


    
    # X_no_age matrix deletes the Age column in the original dataset 
    X_no_age = X
    ID = ['User_ID','Product_ID','Age']
    X_no_age = X_no_age.drop(ID,  axis =1)
    #print(X.shape)

    # split the data into training and test set using sklearn build-in function
    # the test_size = 0.2
    # number of test examples = 107516
    # number of training examples = 430061
    X_train, X_test, y_train, y_test = train_test_split(X_no_age, y, test_size=0.2)

  #  model = KNeighborsClassifier(n_neighbors=5, metric = 'cosine')
  #  model.fit(X_train, y_train)

 #   y_pred = model.predict(X_train)
 #   tr_error = np.mean(y_pred != y_train)

 #   y_pred = model.predict(X_test)
 #   te_error = np.mean(y_pred != y_test)
 #   print("Training error to predict age: %.3f" % tr_error)
 #   print("Testing error to predict age: %.3f" % te_error)
    
    e_depth = 20
    s_depth = 1

    train_errors = np.zeros(e_depth - s_depth)
    test_errors = np.zeros(e_depth - s_depth)

    for i, d in enumerate(range(s_depth, e_depth)):
        print("\nDepth: %d" % d)

        model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        tr_error = np.mean(y_pred != y_train)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

        train_errors[i] = tr_error
        test_errors[i] = te_error

    x_vals = np.arange(s_depth, e_depth)
    plt.title("The effect of tree depth on testing/training error")
    plt.plot(x_vals, train_errors, label="training error")
    plt.plot(x_vals, test_errors, label="testing error")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.legend()

    fname = os.path.join("..", "figs", "trainTest_age.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)
    
    #use decision tree model to predict age
    tree = DecisionTreeClassifier(max_depth=13, criterion='entropy', random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    tr_error = np.mean(y_pred != y_train)
    #Depth: 13
    #Training error: 0.352
    #Testing error: 0.373
    y_pred = tree.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("Training error of predicting occupation: %.3f" % tr_error)
    print("Testing error: %.3f" % te_error)
    
    #use RandomForestClassifier
    model = RandomForestClassifier(criterion="entropy", n_estimators=10, max_features = None)
    model.fit(X_train,y_train)
    print("RandomForest Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("RandomForest Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    
    #use softmaxClassifier to predict occputation
    model = LogisticRegression(C = 1,  fit_intercept=False, solver='lbfgs',
                                   multi_class='multinomial')
    model.fit(X_train,y_train)
    print("LogisticRegression(softmax) Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("LogisticRegression(softmax) Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    # result: 
    # k=10: Training error: 0.526 Testing error: 0.630
    # k=3: Training error: 0.405 Testing error: 0.669
    # k=5: Training error: 0.462 Testing error: 0.650
    
#----------------------------------------------------------------------------------------------------    
    #to predict the occupation
    # Make y matrix to be the occupation
    y_occ = X.values[:,4]
    y_occ = y_occ.astype('int')
    X_occ = X
    ID =['User_ID', 'Product_ID', 'Occupation','Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    X_occ.drop(ID, inplace = True, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X_occ, y_occ, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    tr_error = np.mean(y_pred != y_train)

    y_pred = model.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("Training error of predicting occupation: %.3f" % tr_error)
    print("Testing error of predicting occupation: %.3f" % te_error)

    #use decision tree model
    tree = DecisionTreeClassifier(max_depth=18, criterion='entropy', random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    tr_error = np.mean(y_pred != y_train)
    
    y_pred = tree.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("Training error of predicting occupation: %.3f" % tr_error)
    print("Testing error: %.3f" % te_error)


    #use softmaxClassifier to predict occputation
    model = LogisticRegression(C = 10000,  fit_intercept=False, solver='lbfgs',
                                   multi_class='multinomial')
    model.fit(X_train,y_train)
    print("LogisticRegression(softmax) Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("LogisticRegression(softmax) Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    
    
# use isomap to visualize the data  
    from sklearn.manifold import Isomap
    model = Isomap(n_components = 2)
    ID =['User_ID', 'Product_ID','Product_Category_2', 'Product_Category_3', 'Purchase']
    X_1 = X
    X_1 = X_1.drop(ID, axis = 1)
    fig, ax = plt.subplots()
    Z = model.fit_transform(X_1[:10000])
    ax.scatter(Z[:,0], Z[:,1])
    plt.ylabel('z2')
    plt.xlabel('z1')
    plt.title('ISOMAP with 2components')
    fname = os.path.join("..", "figs", "ISOMAP_with_2_components.png")
    plt.savefig(fname)
    
    model = DBSCAN(eps=1, min_samples=3)
    y = model.fit_predict(Z)
    plt.scatter(Z[:,0], Z[:,1], c=y, cmap="jet", s=5)
    
# clustering the 2 dimensional plot
    model  =  KMeans(n_clusters=5, random_state=0)
    model.fit(Z)
    y = model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1], c=y, cmap="jet")
    plt.ylabel('z2')
    plt.xlabel('z1')
    plt.title('ISOMAP with k_means of 5 clusters')
    plt.show()
    fname = os.path.join("..", "figs", "kmeans.png")
    plt.savefig(fname)
#compress in 3 dimension
    n_compoents = 3
    model = Isomap(n_components = 3)
    Z = model.fit_transform(X_1[:5000])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:,0], Z[:,1], Z[:,2], c='b')
    ax.set_zlabel('z3')
    ax.set_ylabel('z2')
    ax.set_xlabel('z1')
    plt.title('ISOMAP with 3')
    fname = os.path.join("..", "figs", "ISOMAP_with_3_components.png")
    plt.savefig(fname)
 
    
#use PCA to study the data
    ID =['User_ID', 'Product_ID','Product_Category_2', 'Product_Category_3']
    X_1 = X
    X_1 = X_1.drop(ID, axis = 1)
    model = PCA(n_components=3,svd_solver='auto')
    Z = model.fit_transform(X_1[:10000])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:,0], Z[:,1], Z[:,2], c='r')
    ax.set_zlabel('z3')
    ax.set_ylabel('z2')
    ax.set_xlabel('z1')
    plt.title('PCA with 3 components')
    plt.show()
    print(model.explained_variance_ratio_)  
    fname = os.path.join("..", "figs", "PCA.png")
    plt.savefig(fname)
#use pca to study the data 2 componetns
    ID =['User_ID', 'Product_ID','Product_Category_2', 'Product_Category_3']
    X_1 = X
    X_1 = X_1.drop(ID, axis = 1)
    model = PCA(n_components=2 ,svd_solver='auto')
    Z = model.fit_transform(X_1[:100000])
    fig = plt.figure()
    plt.title('PCA with 2 components')
    plt.scatter(Z[:,0], Z[:,1], c='r', cmap="jet", s=5)
    plt.ylabel('z2')
    plt.xlabel('z1')
    fname = os.path.join("..", "figs", "PCA_with_2_components.png")
    print(model.explained_variance_ratio_)  
    plt.savefig(fname)
    #clustering
    ID =['User_ID', 'Product_ID']
    X_1 = X
    X_1 = X_1.drop(ID, axis = 1)
    model = PCA(n_components=2,svd_solver='auto')
    Z = model.fit_transform(X_1[:1000])
    model = DBSCAN(eps=1, min_samples=3)
    y = model.fit_predict(Z)
    plt.scatter(Z[:,0], Z[:,1], c=y, cmap="jet", s=5)
    plt.ylabel('z2')
    plt.xlabel('z1')
    fname = os.path.join("..", "figs", "clustering_from_PCA.png")
    plt.savefig(fname)
    model  =  KMeans(n_clusters=4, random_state=0)
    model.fit(Z)
    y = model.predict(Z)
    plt.scatter(Z[:,0], Z[:,1], c=y, cmap="jet")
    plt.ylabel('z2')
    plt.xlabel('z1')
    plt.title('PCA with NN=%d with k_means from 2 components')
    plt.show()
    
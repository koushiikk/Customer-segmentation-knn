
def full_train_pipeline(df):
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    
    ##train and test split
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 96)
    ##normalizing the data
    X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
    ##training the model
    from sklearn.neighbors import KNeighborsClassifier
    k = 10
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train_norm,y_train)
    ##testing the model
    X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
    y_pred = model.predict(X_test_norm)
    ##accuracy evaluation
    from sklearn import metrics
    train_acc = metrics.accuracy_score(y_train,model.predict(X_train_norm))
    test_acc = metrics.accuracy_score(y_test,y_pred)
    ##Hyperparameter tuning
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
         model = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,y_train)
         y_pred= model.predict(X_test_norm)
         mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)
         std_acc[n-1]=np.std(y_pred==y_test)/np.sqrt(y_pred.shape[0])
    
    #Plots
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
    plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()

    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
   


    return train_acc,test_acc,mean_acc,std_acc
    
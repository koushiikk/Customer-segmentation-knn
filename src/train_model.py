
def full_train_pipeline(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
   
    
    ##train and test split
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 96)
    
    ##normalizing the data
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.astype(float))
    
    ##training the model
    from sklearn.neighbors import KNeighborsClassifier
    k = 10
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train_norm,y_train)
    
    ##testing the model
    X_test_norm = scaler.transform(X_test.astype(float))
    y_pred = model.predict(X_test_norm)
    
    ##accuracy evaluation
    from sklearn import metrics
    train_acc = metrics.accuracy_score(y_train,model.predict(X_train_norm))
    test_acc = metrics.accuracy_score(y_test,y_pred)
    
    ##Hyperparameter tuning
    Ks = 40
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
         model = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,y_train)
         y_pred= model.predict(X_test_norm)
         mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)
         std_acc[n-1]=np.std(y_pred == y_test) / np.sqrt(len(y_test))
        
    ## finding best "K"
    best_k = mean_acc.argmax() +1
    
    ## training again using best K
    best_model = KNeighborsClassifier(n_neighbors = best_k)
    best_model.fit(X_train_norm,y_train)

    ## testing the data with best K
    y_pred_best = best_model.predict(X_test_norm)

    

    cm = confusion_matrix(y_test, y_pred_best)
    print("Confusion Matrix:\n", cm)
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_best))
        
    #Plots
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
    plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.savefig("k_vs_accuracy.png")
    plt.show()

    ##Bar Plot – Gender Distribution
    df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})
    sns.countplot(x='gender', data=df, palette='Set2')
    plt.title('Gender Distribution of Customers')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig("Gender Distribution.png")
    plt.show()

    
    ##Histogram – Age Distribution

    plt.hist(df['age'], bins=15, color='steelblue', edgecolor='black')
    plt.title('Histogram – Age Distribution of Customers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig("Age distribution.png")
    plt.show()

    ##Density Plot – Annual Income

    df['income'].plot(kind='density', color='green')
    plt.title('Density Plot – Annual Income')
    plt.xlabel('Annual Income (k$)')
    plt.savefig("Annual income.png")
    plt.show()

    

    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
   


    return train_acc, test_acc, mean_acc, std_acc, best_k
    
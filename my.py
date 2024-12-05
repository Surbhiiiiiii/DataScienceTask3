# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris_dataset = load_iris()

# Display the feature names of the iris dataset
print("Feature names:", iris_dataset.feature_names)

# Display target names of iris dataset
print("Target names:", iris_dataset.target_names)

# Create a DataFrame from iris dataset
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

# Display some rows of the DataFrame
print(df.head())

# Add target column to the DataFrame
df['target'] = iris_dataset.target
print(df.head())

# Map target values to flower names and add as a new column
df['flower_name'] = df.target.apply(lambda x: iris_dataset.target_names[x])
print(df.head(10))

# Display the first few rows where the target is 1 (versicolor)
print(df[df.target == 1].head())

# Display rows 45 to 54 of DataFrame
print(df[45:55])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# Set labels for x and y axes
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Plot scatter plot for each subset of the DataFrame with different colors and markers
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='^', label='Versicolor')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color="red", marker='*', label='Virginica')

# Add legend to the plot to differentiate between species
plt.legend()

# Set labels for x and y axes to Petal Length and Petal Width
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='^')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="red", marker='*')

# Drop the target and flower_name columns to create feature set
x = df.drop(['target', 'flower_name'], axis='columns')

# Define the target variable
y = df.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize K-Nearest Neighbors classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the K-Nearest Neighbors classifier using the training data
knn.fit(X_train, y_train)

# Evaluate the accuracy of the K-Nearest Neighbors classifier on the test data
print(f"KNN Accuracy: {knn.score(X_test, y_test)}")

# Predict the target values for the test set
y_pred = knn.predict(X_test)

# Compute the confusion matrix to evaluate the accuracy of the classification
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:\n", cm)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 5))
sn.heatmap(cm, annot=True)

# Set the labels for the x and y axes
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Truth', fontsize=15)

# Generate and display the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# DECISION TREE
# Load the iris dataset
iris = load_iris()

# Extract features and target variables
X = iris.data
y = iris.target

# Get feature names and target names
feature_names = iris.feature_names
target_names = iris.target_names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predict the target values for the test set using the trained Decision Tree model
y_pred_dt = dt.predict(X_test)

# Calculate and display the accuracy of the Decision Tree model
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")

# Plot the decision tree with feature and class names
plt.figure(figsize=(15, 10))  # Set the size of the plot
plot_tree(dt, filled=True, feature_names=feature_names, class_names=target_names)  # Plot the decision tree with filled nodes, feature names, and class names
plt.title('Decision Tree', fontsize=30)  # Set the title of the plot with a font size of 30
plt.show()  # Display the plot

# RANDOM FOREST
# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=150, random_state=42)

# Train the Random Forest Classifier
rf.fit(X_train, y_train)

# Predict the target values for the test set using the trained Random Forest model
y_pred_rf = rf.predict(X_test)

# Display Random Forest accuracy
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

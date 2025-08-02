import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocessing import preprocess_data, get_group_labels

def train_models(file_path):
    df = preprocess_data(file_path)
    if df is None:
        return

    # Prepare data
    X = df.drop(columns=['Taxonomy']).T
    y = get_group_labels(df)

    # Filter out unknowns
    X['label'] = y
    X = X[X['label'] != 'Unknown Group']
    y = X.pop('label')

    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Pipeline setup
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Multiple models + hyperparameter grid
    param_grid = [
        {
            'classifier': [LogisticRegression(max_iter=1000)],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        },
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20]
        },
        {
            'classifier': [SVC()],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    ]

    # GridSearch
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    # Best model + performance
    print("Best estimator:", grid.best_estimator_)
    print("Best CV F1 Score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred))


    


if __name__ == "__main__":
    train_models("data/ASD meta abundance.csv")

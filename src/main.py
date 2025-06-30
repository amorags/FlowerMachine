from sklearn.datasets import load_iris
import pandas as pd

def main():
    # Load Iris dataset
    iris = load_iris()

    # Create a pandas DataFrame with features and target
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Print first 5 rows to check
    print("Iris dataset sample:")
    print(df.head())

    # Print dataset description
    print("\nDataset description:")
    print(iris.DESCR)

if __name__ == "__main__":
    main()

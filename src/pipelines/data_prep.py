import pandas as pd
import argparse

def main(input_data, output_data):

    df = pd.read_csv(input_data)

    print("Original shape:", df.shape)

    # Drop ID column
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop leakage column
    if "Payment Delay" in df.columns:
        df = df.drop("Payment Delay", axis=1)

    print("Cleaned shape:", df.shape)

    df.to_csv(output_data, index=False)
    print("✅ Cleaned data saved:", output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_data", type=str)

    args = parser.parse_args()

    main(args.input_data, args.output_data)
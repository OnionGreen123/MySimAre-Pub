import pandas as pd


def main() -> None:
    input_path = "train_fixed.csv"
    output_path = "train_fixed_level_4_5.csv"

    df = pd.read_csv(input_path)
    df_filtered = df[df["level"].isin(["Level 4", "Level 5"])]
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved {len(df_filtered)} rows to {output_path}")


if __name__ == "__main__":
    main()


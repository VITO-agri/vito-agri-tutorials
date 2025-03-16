import json
import pandas as pd
from pathlib import Path


def openeo_timeseries_json_to_df(
    json_file: Path, sample_ids: list[str], variable_names: list[str]
):
    """Convert OpenEO timeseries JSON file to DataFrame.
    Parameters
    ----------
    json_file : Path
        Path to JSON file.
    sample_ids : list[str]
        unique id's of the samples
    variable_names : list[str]
        unique names of the variables
    Returns
    -------
    DataFrame
        DataFrame containing timeseries data.
    """

    # Load JSON file
    with open(json_file, "r") as f:
        json_data = json.load(f)

    # Convert JSON into a DataFrame
    df = pd.DataFrame.from_dict(json_data, orient="index")

    # Expand each list into separate columns (variables first, then samples)
    df = df.apply(
        lambda x: pd.DataFrame(
            x.tolist(), columns=[f"Var{i+1}" for i in range(len(x[0]))]
        ),
        axis=1,
    )

    # Convert the nested structure into a proper DataFrame
    df = pd.concat(df.tolist(), keys=df.index)

    # Rename the index levels
    df.index.names = ["Timestamp", "Sample"]

    # Convert timestamp index to datetime
    df = (
        df.reset_index()
        .sort_values(["Timestamp", "Sample"])
        .set_index(["Timestamp", "Sample"])
    )

    df.columns = variable_names

    # Rename the sample index
    df = df.reset_index()  # Convert MultiIndex to columns
    df["Sample"] = df["Sample"].map(lambda x: sample_ids[x])  # Apply sample names
    df = df.set_index(["Timestamp", "Sample"])  # Set index back to MultiIndex

    return df


def multi_index_df_to_single_index(df: pd.DataFrame, bandname: str) -> pd.DataFrame:

    all_timestamps = df.index.get_level_values("Timestamp").unique()
    df_pivoted = df.pivot_table(index="Timestamp", columns="Sample", values=bandname)
    df_pivoted = df_pivoted.reindex(all_timestamps)
    df_pivoted = df_pivoted.rename_axis(None, axis=0)  # Remove row index name
    df_pivoted.index = pd.to_datetime(df_pivoted.index)

    return df_pivoted

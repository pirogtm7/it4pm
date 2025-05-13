from datasets.arrow_dataset import Dataset
import pandas as pd
import pickle
import ast

def remove_duplicates(pair_df):
    columns = ["revision_id", "model_id", "unique_activities"]
    if "trace" in pair_df.columns and "prefix" not in pair_df.columns:
        columns.append("trace")
    if "eventually_follows" in pair_df.columns:
        columns.append("eventually_follows")
    if "prefix" in pair_df.columns:
        columns.append("prefix")
        columns.append("next")
    pair_df = pair_df.drop_duplicates(subset=columns)
    return pair_df


def setify(x: str):
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_


def split_by_model(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    df["num_unique_activities"] = df["unique_activities"].apply(len)

    # only keep rows with at least 2 activities
    df = df[df["num_unique_activities"] > 1]

    with open(f"datasets/train_val_test.pkl", "rb") as file:
        train_ids, val_ids, test_ids = pickle.load(file)
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]

    return train_df, val_df, test_df


def load_trace_data() -> Dataset:    
    trace_df: pd.DataFrame = pd.read_csv("datasets/T_SAD.csv")
    
    trace_df = remove_duplicates(trace_df)

    # using a set ensures that each activity is listed only once, removing duplicates automatically
    trace_df.unique_activities = trace_df.unique_activities.apply(setify)

    # invert labels because the model will predict whether the trace is correct, and not wrong (=anomalous)
    trace_df["ds_labels"] = (~trace_df["anomalous"]).astype(str)  # Convert to str for labels

    # parse trace into literals for correct processing
    trace_df["trace"] = trace_df["trace"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    columns = ["model_id", "revision_id", "unique_activities", "trace", "ds_labels"]
    trace_df = trace_df.loc[:, columns]

    train_df, val_df, test_df = split_by_model(trace_df)
    
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def load_pairs_data() -> Dataset:
    pairs_df: pd.DataFrame = pd.read_csv("datasets/A_SAD.csv")

    pairs_df = remove_duplicates(pairs_df)

    # using a set ensures that each activity is listed only once, removing duplicates automatically
    pairs_df.unique_activities = pairs_df.unique_activities.apply(setify)

    # invert labels
    pairs_df["ds_labels"] = (~pairs_df["out_of_order"]).astype(str)

    # parse eventually_follows into literals for correct processing
    pairs_df["eventually_follows"] = pairs_df["eventually_follows"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    columns = ["model_id", "revision_id", "unique_activities", "eventually_follows", "ds_labels"]
    pairs_df = pairs_df.loc[:, columns]

    train_df, val_df, test_df = split_by_model(pairs_df)
    
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def load_next_activity_data() -> Dataset:
    next_activity_df: pd.DataFrame = pd.read_csv("datasets/S_NAP.csv")

    next_activity_df = remove_duplicates(next_activity_df)

    # using a set ensures that each activity is listed only once, removing duplicates automatically
    next_activity_df.unique_activities = next_activity_df.unique_activities.apply(setify)

    # removes rows where the next column contains [END], indicating the process has reached its conclusion and there is no subsequent activity to predict.
    next_activity_df = next_activity_df[~(next_activity_df.next == "[END]")]

    # remove rows where "None" activity is present as it's not an actual activity
    next_activity_df = next_activity_df[next_activity_df["unique_activities"].apply(lambda x: "None" not in x and None not in x)]

    # parse prefix into literals for correct processing
    next_activity_df["prefix"] = next_activity_df["prefix"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    columns = ["model_id", "revision_id", "unique_activities", "trace", "prefix", "next"]
    next_activity_df = next_activity_df.loc[:, columns]

    train_df, val_df, test_df = split_by_model(next_activity_df)

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )

    

def load_discovery_data() -> Dataset:
    discovery_df: pd.DataFrame = pd.read_csv("datasets/S-PMD.csv")

    discovery_df = remove_duplicates(discovery_df)

    # using a set ensures that each activity is listed only once, removing duplicates automatically
    discovery_df.unique_activities = discovery_df.unique_activities.apply(setify)

    columns = ["model_id", "revision_id", "unique_activities", "dfg", "pt"]
    discovery_df = discovery_df.loc[:, columns]

    train_df, val_df, test_df = split_by_model(discovery_df)

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )

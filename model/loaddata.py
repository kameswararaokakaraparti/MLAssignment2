import pandas as pd
from typing import List

def load_data_from_csv(
    file_path: str,
) -> pd.DataFrame:

    df = pd.read_csv(file_path)

    df = df.drop(columns=['Sex', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'Stroke'])

    return df
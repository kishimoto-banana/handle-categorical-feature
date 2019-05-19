from imblearn.under_sampling import RandomUnderSampler


def train_test_split(df, test_rate=0.2):

    test_splited_index = int(len(df) * test_rate)
    df_train = df[:len(df) - test_splited_index]
    df_test = df[-test_splited_index:]

    return df_train, df_test


def fillna_integer_feature(df, integer_columns):

    df[integer_columns] = df[integer_columns].fillna(-1.0)
    return df


def fillna_categorical_feature(df, categorical_columns):

    df[categorical_columns] = df[categorical_columns].fillna('')
    return df


def under_sampling(X, y, random_state: int = 42):

    num_pos = y.sum()
    sampler = RandomUnderSampler(ratio={
        0: num_pos,
        1: num_pos
    },
                                 random_state=random_state)

    sampler.fit_sample(X, y)
    sampled_indicies = sampler.sample_indices_

    return sampled_indicies

import pandas as pd
from sklearn.utils import resample


def vy_training_augmentation(process_df):
    process_df = process_df.copy()
    a_crab_df = process_df[process_df["original_material_name"]
                           == "ปูทั้งตัว A"]
    c_crab_df = process_df[process_df["original_material_name"]
                           == "ปูทั้งตัว C"]
    only_small_crab_df = process_df[process_df["original_material_name"]
                                    == "ปูทั้งตัว จิ๋ว"]
    only_loss_crab_df = process_df[process_df["original_material_name"]
                                   == "ปูทั้งตัว โพรก"]
    small_crab_df = process_df[process_df["original_material_name"]
                               == "ปูจิ๋ว และโพรก"]

    # Compare the amount of each crab type rows
    a_crab_size = a_crab_df.shape[0]
    c_crab_size = c_crab_df.shape[0]
    small_crab_size = small_crab_df.shape[0]
    only_small_size = only_small_crab_df.shape[0]
    only_loss_size = only_loss_crab_df.shape[0]

    # Find the maximum size of the crab type
    max_size = max(a_crab_size, c_crab_size, small_crab_size,
                   only_small_size, only_loss_size)

    # Calculate the number of rows to add for each crab type
    a_crab_add = max_size * 2 - a_crab_size
    c_crab_add = max_size * 2 - c_crab_size
    small_crab_add = max_size * 2 - small_crab_size
    only_small_crab_add = max_size * 2 - only_small_size
    only_loss_crab_add = max_size * 2 - only_loss_size

    # Add rows to each crab type
    if a_crab_size > 0:
        a_crab_augmented = resample(a_crab_df, n_samples=a_crab_add)
    else:
        a_crab_augmented = pd.DataFrame()
    if c_crab_size > 0:
        c_crab_augmented = resample(c_crab_df, n_samples=c_crab_add)
    else:
        c_crab_augmented = pd.DataFrame()
    if small_crab_size > 0:
        small_crab_augmented = resample(
            small_crab_df, n_samples=small_crab_add)
    else:
        small_crab_augmented = pd.DataFrame()
    if only_small_size > 0:
        only_small_crab_augmented = resample(
            only_small_crab_df, n_samples=only_small_crab_add)
    else:
        only_small_crab_augmented = pd.DataFrame()
    if only_loss_size > 0:
        only_loss_crab_augmented = resample(
            only_loss_crab_df, n_samples=only_loss_crab_add)
    else:
        only_loss_crab_augmented = pd.DataFrame()

    # Concatenate the augmented dataframes
    a_crab_df_combined = pd.concat([a_crab_df, a_crab_augmented])
    c_crab_df_combined = pd.concat([c_crab_df, c_crab_augmented])
    small_crab_df_combined = pd.concat([small_crab_df, small_crab_augmented])

    only_small_crab_df_combined = pd.concat(
        [only_small_crab_df, only_small_crab_augmented])
    only_loss_crab_df_combined = pd.concat(
        [only_loss_crab_df, only_loss_crab_augmented])

    # New Process Dataframe
    process_df = pd.concat(
        [a_crab_df_combined, c_crab_df_combined,
         small_crab_df_combined, only_small_crab_df_combined,
         only_loss_crab_df_combined
         ])

    return process_df

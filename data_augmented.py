import pandas as pd

def augment_datetime_months(df, target_column='pip_class', specific_months=[9,10,11,12,1],
                           augmentation_factor_2019=6,
                           augmentation_factor_2020=6,
                           augmentation_factor_2021=7,
                           augmentation_factor_2022=7,
                           augmentation_factor_2023=9,
                           augmentation_factor_2024=4):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure datetime is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Separate the features and target
    X = df.drop(columns=[target_column, 'datetime', 'Year'])
    y = df[target_column]

    # Identify indices for each year
    year_indices = {year: df[df['Year'] == year].index for year in range(2019, 2025)}

    augmented_data = []
    augmented_targets = []

    def augment_samples(indices, augmentation_factor):
        for idx in indices:
            original_sample = X.loc[idx]
            original_datetime = df.loc[idx, 'datetime']
            original_year = df.loc[idx, 'Year']
            original_target = df.loc[idx, target_column]

            # Only augment if the original month is in specific_months and target is 1
            if original_datetime.month in specific_months and original_target == 1:
                for i in range(augmentation_factor - 1):  # -1 because we already have the original sample
                    synthetic_sample = original_sample.copy()

                    # Create a new datetime by cycling through specific months
                    new_month = specific_months[(specific_months.index(original_datetime.month) + i + 1) % len(specific_months)]
                    new_datetime = original_datetime.replace(month=new_month, day=1)

                    synthetic_sample['datetime'] = new_datetime
                    synthetic_sample['Year'] = original_year

                    augmented_data.append(synthetic_sample)
                    augmented_targets.append(original_target)

    # Augment data for each year based on specified factors
    for year, factor in [
        (2019, augmentation_factor_2019),
        (2020, augmentation_factor_2020),
        (2021, augmentation_factor_2021),
        (2022, augmentation_factor_2022),
        (2023, augmentation_factor_2023),
        (2024, augmentation_factor_2024)
    ]:
        augment_samples(year_indices[year], factor)

    # Create a DataFrame from the augmented data
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        augmented_df[target_column] = augmented_targets
        augmented_df['datetime'] = pd.to_datetime(augmented_df['datetime'])

        # Combine the original and augmented data
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
    else:
        combined_df = df.copy()

    # Sort by datetime to maintain chronological order
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

    return combined_df


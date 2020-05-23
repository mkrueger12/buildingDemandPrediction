from functions import *
import boto3

# Data Parameters
s3 = boto3.client('s3', aws_access_key_id='AKIA6DK5LBRLFPRTV4C5',
                  aws_secret_access_key='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')

aws_bucket = 'energydataaps'
weather_file = 'BKF.txt'
energy_file = 'combined_data.csv'

# Collect data
weather = weatherData(aws_bucket, weather_file, s3)
energy = energyDataAPS(aws_bucket, energy_file, s3)

# Standardize Date/Time
print('Creating DT index')
weather.rename({'valid': 'date'}, axis=1, inplace=True)

weather['date'] = pd.to_datetime(weather['date'])

weather = weather.dropna()

datetimeindex(weather)  # Add datetime index to weather df

energy.rename({'date & time': 'date', 'usage [kw]': 'usage'}, axis=1, inplace=True)

energy['date'] = pd.to_datetime(energy['date'])

# join data frames
print('Joining')
weather['date'] = weather['date'].apply(lambda x: x.strftime("%Y-%m-%d %H"))

energy['date'] = energy['date'].apply(lambda x: x.strftime("%Y-%m-%d %H"))

comb_data = pd.merge(energy, weather, on='date', how='inner')

# Remove First 9 Rows
comb_data = comb_data.dropna().drop_duplicates()

# Write to s3
print('Writing to s3')
toS3(aws_bucket, comb_data, s3, 'combined_data_clean.csv')

import pandas as pd
import io
from io import StringIO

def weatherData(bucket, weather_file, s3):

    ''' Return a dataframe of a weather file from
    https://mesonet.agron.iastate.edu/request/download.phtml?network=MD_ASOS.
     Must define s3 client. '''

    print('Weather Data Downloading')
    obj = s3.get_object(Bucket=bucket, Key=weather_file)
    weather = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print('Weather Data Downloaded')
    weather = weather[['valid', 'tmpf', 'dwpf', 'relh', 'sknt', 'alti', 'skyc1']]

    return weather

# Stream Energy Data

def energyDataAPS(bucket, energy_file, s3):

    ''' Returns a dataframe report of energy data from APS.
     Must define s3 client. '''

    print('Energy Data Downloading')
    obj = s3.get_object(Bucket = bucket, Key = energy_file)
    energy = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print('Energy Data Downloaded')

    return energy

def datetimeindex(df):
    """
    Creates time series features from datetime index
    """
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X

def toS3(bucket, df, s3, filename):

    ''' Writes a pandas df to s3 as csv '''
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=filename)
    print('s3 Upload Complete')

    return
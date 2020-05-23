import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import boto3
import io

# Data Parameters
s3 = boto3.client('s3', aws_access_key_id='XXXXXXXXX',
                  aws_secret_access_key='XXXXXX')

aws_bucket = 'energydataaps'
data_file = 'combined_data_clean.csv'

obj = s3.get_object(Bucket=aws_bucket, Key=data_file)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

# Select Records
df = df[df['year'] < 2020]

df = df.drop(columns=['Unnamed: 0', 'date', 'year'])

dataset = df[df['usage'] > 5]

# Change data types
dataset.isna().sum()
dataset = df.dropna()
dataset.info()

# One-hot encode
dummy = pd.get_dummies(df['type'], prefix='type')
dataset = pd.concat([dataset, dummy], axis=1)

dummy = pd.get_dummies(df['skyc1'], prefix='sky')
dataset = pd.concat([dataset, dummy], axis=1)

dummy = pd.get_dummies(df['name'], prefix='name')
dataset = pd.concat([dataset, dummy], axis=1)

dataset = dataset.drop(columns=['type', 'skyc1', 'name'])

# Split Data
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('usage')
train_stats = train_stats.transpose()
train_stats

# Split Features and Labels
train_labels = train_dataset.pop('usage')
test_labels = test_dataset.pop('usage')

# Normalize
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Create and Train Model
def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()

model.summary()


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))


lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train Model
EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

print('Model Running')

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=1,
  callbacks=[lr, early_stop])

# Visualize
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric="mae") # MAE
plt.ylim([0, 50])
plt.ylabel('MAE [kW]')

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing Set Mean Abs Error: {:5.2f} kW".format(mae))

# Predict
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [kW]')
plt.ylabel('Predictions [kW]')
lims = [0, 500]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

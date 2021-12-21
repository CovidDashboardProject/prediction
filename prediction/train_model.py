
from src import *


#get the data
df = get_data()
date_time = pd.to_datetime(df.pop('Date'), format='%d/%m/%Y')

timestamp_s = date_time.map(pd.Timestamp.timestamp)

#split
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):int(n*0.9)]
test_df = df[int(n*0.90):]

num_features = df.shape[1]

#normalize
train_mean = train_df.mean()
train_std = train_df.std()


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#create data window
OUT_STEPS = 21
multi_window = WindowGenerator(input_width=30,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df = train_df,
                               val_df = val_df,
                               test_df = test_df)

#create model
feedback_model = FeedBack(units=2048, out_steps=OUT_STEPS)

prediction, state = feedback_model.warmup(multi_window.example[0])


#train
compile_and_fit(feedback_model, multi_window,MAX_EPOCHS = 500,patience=5)
#open street Routing Model

#Import Libraries
import pandas as pd
import httpx

#host and port
osrm_host = "http://127.0.0.1"
osrm_port = 5000

#read file
df = pd.read_csv("./selected_data.csv")

df.isnull().sum()
df.info()

#values to numeric
df = df[pd.to_numeric(df['pickup_longitude'], errors='coerce')]
df = df[pd.to_numeric(df['pickup_latitude'], errors='coerce')]
df = df[pd.to_numeric(df['dropoff_longitude'], errors='coerce')]
df = df[pd.to_numeric(df['dropoff_latitude'], errors='coerce')]

#change type values
df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']] = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].astype(float)

#definde invalid data (data just lat:-90_90 & lon:-180_180)
invalid_data = df[
    (df['pickup_longitude'] < -180) | (df['pickup_longitude'] > 180) |
    (df['pickup_latitude'] < -90) | (df['pickup_latitude'] > 90) |
    (df['dropoff_longitude'] < -180) | (df['dropoff_longitude'] > 180) |
    (df['dropoff_latitude'] < -90) | (df['dropoff_latitude'] > 90)
]

print(f"count data invalid: {len(invalid_data)}")
print(invalid_data)

#data just lat:-90_90 & lon:-180_180
df = df[
    (df['pickup_longitude'].between(-180, 180)) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_longitude'].between(-180, 180)) &
    (df['dropoff_latitude'].between(-90, 90))
]    #118 rows has invalid data

#values invalid example: '/n' , '//' , ...
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

#request osrm for duration and distance
def get_response(lon_src, lat_src, lon_dest, lat_dest):
    request_url = f"{osrm_host}:{osrm_port}/route/v1/driving/{lon_src},{lat_src};{lon_dest},{lat_dest}?overview=false"
    try:
        response = httpx.get(request_url)
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                duration = route['duration']  # At (scecond)
                distance = route['distance']  # Al (meter)
                return duration, distance
    except Exception as e:
        print(f"Error fetching route: {e}")
    return None, None

#get values for columns route duration and route distance
df[['route_duration', 'route_distance']] = df.apply(
    lambda row: pd.Series(get_response(
        row['pickup_longitude'], row['pickup_latitude'],
        row['dropoff_longitude'], row['dropoff_latitude']
    )),
    axis=1
)
#save response to csv
df.to_csv("duration_distance.csv", index=False)
print('successfully saved')

#After get 2 columns distance and duration, write code for prediction travel duration
#just one file duration_distance.csv and totall file concanet and then prediction

#import libreries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from skopt.space import Integer

#read file 1 (duration and distance)
df1 = pd.read_csv("./duration_distance.csv")

#read file 2 (taxi_data)
df2 = pd.read_csv("./taxi_data.csv")

#concanet 2 files
df = pd.concat([df1,df2] , axis=1)

#save file concat to csv
df = df.to_csv("travel_duration_taxi.csv" , index=False)

#read file concat
df = pd.read_csv("./travel_duration_taxi.csv")
df.isnull().sum()
df.info()
df.describ()

##remove nan values and filling by KNN
imp = KNNImputer(missing_values=np.nan , n_neighbors=3)
df[['dropoff_latitude','dropoff_longitude']] = imp.fit_transform(df[['dropoff_latitude','dropoff_longitude']])
df.isnull().sum()

#change dtype time
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

#Extracting temporal features from pickup_datetime
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['hour']= df['pickup_datetime'].dt.hour
df['minute']= df['pickup_datetime'].dt.minute
df['weekday'] = df['pickup_datetime'].dt.weekday
df['dayofweek']= df['pickup_datetime'].dt.dayofweek
df['is_weekend'] = df['pickup_datetime'].dt.weekday >= 5

##Standard Scaling
scaler = StandardScaler()
df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]=scaler.fit_transform(df[['pickup_latitude','pickup_longitude',

#Data analysis and Exploration
                                                                                                           'dropoff_latitude','dropoff_longitude']])
#VONDER ID
fig , axes = plt.subplots(nrows=1 , ncols = 2 , figsize=(12,5))
ax = df['vendor_id'].value_counts().plot(kind='bar', title = 'vonders',ax=axes[0] , color=('blue',(1,0.5,0.13)))
df['vendor_id'].value_counts().plot(kind='pie', title='vonders' , ax=axes[1])
ax.set_ylabel('count')
ax.set_xlabel('vonder ID')
fig.tight_layout()

#Travel Duration
plt.Figure(figsize=(25,5))
sns.boxplot(df.route_duration)
plt.show()

#Duration
df.route_duration.groupby(pd.cut(df.route_duration , np.arange(1,7200,600))).count().plot(kind='barh' , figsize=(18,5))
plt.title('trip duration')
plt.xlabel('trip count')
plt.ylabel('trip duration')
plt.show()

#Distance
plt.figure(figsize=(25,5))
sns.boxplot(df.route_distance)
plt.show()

#duration & distance
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='route_distance', y='route_duration', alpha=0.5)
plt.title("Travel Duration vs Distance")
plt.xlabel("Distance")
plt.ylabel("Travel Duration")
plt.show()

#travel duration by hour
sns.boxplot(data=df, x='hour', y='route_duration')
plt.title("Travel Duration by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Travel Duration")
plt.show()

#travel duration by day of week
sns.boxplot(data=df, x='dayofweek', y='route_duration')
plt.title("Travel Duration by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Travel Duration (seconds)")
plt.show()

#distance by hour
sns.boxplot(data=df, x='hour', y='route_distance')
plt.title("Distance by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Distance (km)")
plt.show()

#heatmap
correlation_matrix = df[['route_duration', 'route_distance', 'hour', 'dayofweek','year','month','weekday','is_weekend','passenger_count','rate_code']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Dataset
x = df[['year','month','day','hour','minute','weekday','dayofweek','is_weekend','route_distance',
        'passenger_count','rate_code','fare_amount','surcharge','mta_tax','tip_amount',
        'tolls_amount','total_amount']]
y = df['route_duration'].values.ravel()

# Split dataset into training and testing sets
x_train , x_test,y_train,y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

# Initialize Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# Perform cross-validation
scores = cross_val_score(dt_model, x_train, y_train, cv=5 ,scoring='r2')

print(f'Cross-validation scores: {scores}')
print(f'Mean CV Score: {scores.mean():.2f}') 
print(f'CV Score Standard Deviation: {scores.std():.2f}')

dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error on Test Set: {mse:.2f}')

# Define the parameters to search
param_ranges= {
    'max_depth': Integer(5, 15), 
    'min_samples_split': Integer(2, 10),  
    'min_samples_leaf': Integer (1, 4),
     }
# Initialize BayesSearchCV
bayes_search = BayesSearchCV(estimator=dt_model, search_spaces=param_ranges, 
                                n_iter=5, cv=5, random_state=42,
                                n_jobs=-1, scoring = 'r2',error_score=np.nan)

# Fit the model
bayes_search.fit(x_train, y_train)

# Best parameters and score
print(f'Best parameters: {bayes_search.best_estimator_.get_params()}')
print(f'Best score: {bayes_search.best_score_}')

# Make predictions on the test set
y_pred = bayes_search.predict(x_test)

# Calculate mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
dt_score = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2: {dt_score:.2f}')

#Prediced and Actual values chart
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
plt.title('Actual vs Predicted', fontsize=16)
plt.xlabel('Actual Travel Duration', fontsize=14)
plt.ylabel('Predicted Travel Duration', fontsize=14)
plt.grid(True)
plt.show()

#Model Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_regression = RandomForestRegressor()
rf_regression = rf_regression.fit(x_train , y_train)
y_pred = rf_regression.predict(x_test)

#parameter for random forest
rf_param = {
    'max_depth': Integer(5, 15), 
    'min_samples_split': Integer(2, 10),  
    'min_samples_leaf': Integer (1, 4),

}

# Initialize BayesSearchCV
bys_search = BayesSearchCV(estimator=rf_regression, search_spaces=rf_param, 
                                n_iter=5, cv=5, random_state=42,
                                n_jobs=-1, scoring = 'r2',error_score=np.nan)

bys_search.fit(x_train, y_train)

print(f'Best parameters: {bys_search.best_estimator_.get_params()}')
print(f'Best score: {bys_search.best_score_}')

y_pred = bys_search.predict(x_test)

#mean squared error
mse = mean_squared_error(y_test, y_pred)
rf_score= r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2: {rf_score:.2f}')

#Model XGBoost
from xgboost import XGBRegressor

xg_model = XGBRegressor()
xg_model = xg_model.fit(x_train , y_train)
xg_predict = xg_model.predict(x_test)

xg_score = r2_score(y_test , xg_predict)
print(xg_score)

#Model Compration
r2 = [xg_score , rf_score , dt_score]
cm = pd.DataFrame({'Accuracy':r2})

label = ['XGBoost','Random Forest','Decision Tree']
fig , axes = plt.subplots(nrows = 1 , ncols = 2 , figsize=(12,5))
ax = cm['Accuracy'].plot(kind='bar' , title='Accuracy', ax=axes[1])
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(label)
fig.tight_layout()








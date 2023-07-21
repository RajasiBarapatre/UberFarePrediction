import streamlit as st
import pandas as pd
import folium
from geopy import distance
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from math import radians,sin, cos, sqrt, asin
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

st.title("UBER FARE PREDICTION")
# Load the data from CSV file
data = pd.read_csv(r"C:\Users\91846\OneDrive\Desktop\Uber\Uber_Fare.csv")
# Extract latitude and longitude columns from the data
lat = data['pickup_latitude']
lon = data['pickup_longitude']
# Create a DataFrame with latitude and longitude columns
locations = pd.DataFrame({'latitude': lat, 'longitude': lon})
#button_style = "background-color: blue; color: white;"
home=st.sidebar.button("HOME")
if(home==True):
# Plot the locations on a map using Streamlit's map function
  st.map(locations)
# Get user input for latitude and longitude
lat1 = st.sidebar.number_input("Start Latitude:",format="%.5f")
lon1 = st.sidebar.number_input("Start Longitude :",format="%.5f")
lat2 = st.sidebar.number_input("Destination Latitude:",format="%.5f")
lon2 = st.sidebar.number_input("Destination Longitude:",format="%.5f")
passenger = st.sidebar.number_input("Passenger Count:",step=1)
car=st.sidebar.selectbox( "Car Model",('Hatchback','Sedan','SUV'))
# Calculate the distance between the two points using geopy.distance
coord1 = (lat1, lon1)
coord2 = (lat2, lon2)
dist = distance.distance(coord1, coord2).km
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%H:%M:%S")
cur=0
if '00:00:00' < currentTime < '07:00:00':
  cur=1
elif '07:00:00'<= currentTime < '12:00:00':
  cur=2
elif '12:00:00' <=currentTime < '17:00:00':
  cur=3
else:
  cur=4
speed=0
if cur==1:
  speed=70
elif cur==2:
  speed=40
elif cur==3:
  speed=50
else:
  speed=40
time_needed = dist / speed
time=[]
for i in range(0,1999):
  temp=data['pickup_datetime'][i].split(' ')[1]
  time.append(temp)
data['Time']=time
data.drop(['key'], axis=1, inplace= True)
# convert 'Time' column to datetime objects
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
for i in range(len(data)):
  if datetime.strptime('00:00:00', '%H:%M:%S') < data.loc[i, 'Time'] < datetime.strptime('06:00:00', '%H:%M:%S'):
    data.loc[i, 'Time'] = 0
  elif datetime.strptime('06:00:00', '%H:%M:%S') <= data.loc[i,'Time'] < datetime.strptime('12:00:00', '%H:%M:%S'):
    data.loc[i, 'Time'] = 1
  elif datetime.strptime('12:00:00', '%H:%M:%S') <=data.loc[i,'Time'] < datetime.strptime('17:00:00', '%H:%M:%S'):
    data.loc[i, 'Time'] = 2
  else:
    data.loc[i, 'Time'] = 3
# convert 'Time' column back to integers
data['Time'] = data['Time'].astype(int)
data.drop(['pickup_datetime'], axis=1, inplace= True)
pickup=[]
for i in range(0,1999):
p=[data['pickup_latitude'][i],data['pickup_longitude'][i]]
pickup.append(p)
dropoff=[]
for i in range(0,1999):
p=[data['dropoff_latitude'][i],data['dropoff_longitude'][i]]
dropoff.append(p)
data['Drop_off']=dropoff
data['Pick_up']=pickup
# Haversine Formula
distance=[]

R = 6371.0
for x in range(0,1999):
  p=radians(data['Pick_up'][x][0])
  q=radians(data['Pick_up'][x][1])
  r=radians(data['Drop_off'][x][0])
  s=radians(data['Drop_off'][x][1])
  dlon = q - s
  dlat = p - r
  a = (sin(dlat/2))**2 + cos(p) * cos(r) * (sin(dlon/2))**2
  c = 2 * asin(sqrt(a))
  dis= R * c
  distance.append(dis)
data['Distance']=distance

option=st.selectbox("Select Model",('','Random Forest Regressor','Multiple Regression','Elastic Net Regression'))
if option == 'Random Forest Regressor':
  #DISPLAY
  # Display the distance between the two points
  st.write("Current Time",currentTime)
  st.write("Distance between the two points:", dist, "km")
  if time_needed >= 1:
    st.write("Time needed for travel:",time_needed, "hours")
  elif time_needed <1:
    st.write("Time needed for travel:",time_needed*60, "minutes")
  # Create a map using folium
  m = folium.Map(location=[lat1, lon1], zoom_start=12)
  # Add markers for the two points
  folium.Marker(location=[lat1, lon1], tooltip="Point 1",icon=folium.Icon(color="green")).add_to(m)
  folium.Marker(location=[lat2, lon2], tooltip="Point 2",icon=folium.Icon(color="red")).add_to(m)
  # Add a line connecting the two points
  folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],color='blue').add_to(m)
  # Display the map
  folium_static(m)
  X = data[['Distance','Time','passenger_count']]
  Y = data['fare_amount']
  X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.25, shuffle= True)
  random=RandomForestRegressor()
  
  random.fit(X_train,Y_train)
  Y_pred=random.predict(X_test)
  mse = mean_squared_error(Y_test, Y_pred)
  r2 = r2_score(Y_test, Y_pred)
  rmse_rf = np.sqrt(mse) / np.mean(Y_test)
  # Print results
  st.write("Root Mean Squared Error: {:.2f}".format(rmse_rf ))
  sample=np.array([[dist,time_needed,passenger]])
  predict1=random.predict(sample)
  if car=='Hatchback':
    st.write("Fare amount= $",predict1[0])
  elif car=='Sedan':
    st.write("Fare amount= $",predict1[0]+predict1[0]*0.05)
  elif car=='SUV':
    st.write("Fare amount= $",predict1[0]+predict1[0]*0.1)

elif option=='Multiple Regression':
  st.write("Current Time",currentTime)
  # Display the distance between the two points
  st.write("Distance between the two points:", dist, "km")
  if time_needed >= 1:
    st.write("Time needed for travel:",time_needed, "hours")
  elif time_needed <1:
    st.write("Time needed for travel:",time_needed*60, "minutes")
  # Create a map using folium
  m = folium.Map(location=[lat1, lon1], zoom_start=12)
  # Add markers for the two points
  folium.Marker(location=[lat1, lon1], tooltip="Point 1",icon=folium.Icon(color="green")).add_to(m)
  folium.Marker(location=[lat2, lon2], tooltip="Point 2",icon=folium.Icon(color="red")).add_to(m)
  # Add a line connecting the two points
  folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],color='blue').add_to(m)
  # Display the map
  folium_static(m)
  a = data[['Distance', 'Time','passenger_count']]
  b=data['fare_amount']
  a_train,a_test,b_train,b_test=train_test_split(a, b,
  test_size=0.25, shuffle= True)
  lin=LinearRegression()
  
  lin.fit(a_train,b_train)
  a_pred=lin.predict(a_test)
  # cd=lin.score(a_train,b_train)
  # st.write("Coefficient of Determination : ",cd)
  mse = mean_squared_error(b_test, a_pred)
  r2 = r2_score(b_test, a_pred)
  rmse_mr = np.sqrt(mse) / np.mean(b_test)
  # Print results
  st.write("Root Mean Squared Error: {:.2f}".format(rmse_mr ))
  sample=np.array([[dist,time_needed,passenger]])
  predict2=lin.predict(sample)
  if car=='Hatchback':
    st.write("Fare amount= $",predict2[0])
  elif car=='Sedan':
    st.write("Fare amount= $",predict2[0]+predict2[0]*0.05)
  elif car=='SUV':
    st.write("Fare amount= $",predict2[0]+predict2[0]*0.1)

elif option =='Elastic Net Regression':
  st.write("Current Time",currentTime)
  # Display the distance between the two points
  st.write("Distance between the two points:", dist, "km")
  if time_needed >= 1:
    st.write("Time needed for travel:",time_needed, "hours")
  elif time_needed <1:
    st.write("Time needed for travel:",time_needed*60, "minutes")
  # Create a map using folium
  m = folium.Map(location=[lat1, lon1], zoom_start=12)
  # Add markers for the two points
  folium.Marker(location=[lat1, lon1], tooltip="Point 1",icon=folium.Icon(color="green")).add_to(m)
  folium.Marker(location=[lat2, lon2], tooltip="Point 2",icon=folium.Icon(color="red")).add_to(m)
  # Add a line connecting the two points
  folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],color='blue').add_to(m)
  # Display the map
  folium_static(m)
  P= data[['Distance', 'Time','passenger_count']]
  Q=data['fare_amount']
  # P, Q = make_regression(n_features=2, random_state=0)
  regr = ElasticNet()
  P_train,P_test,Q_train,Q_test=train_test_split(P, Q,test_size=0.25, shuffle= True)
  regr.fit(P_train, Q_train)
  
  Q_pred=regr.predict(P_test)
  mse = mean_squared_error(Q_test, Q_pred)
  r2 = r2_score(Q_test, Q_pred)
  rmse_en = np.sqrt(mse) / np.mean(Q_test)
  # Print results
  st.write("Root Mean Squared Error: {:.2f}".format(rmse_en))
  sample=np.array([[dist,time_needed,passenger]])
  predict3=regr.predict(sample)
  if car=='Hatchback':
    st.write("Fare amount= $",predict3[0])
  elif car=='Sedan':
    st.write("Fare amount= $",predict3[0]+predict3[0]*0.05)
  elif car=='SUV':
    st.write("Fare amount= $",predict3[0]+predict3[0]*0.1)

graphs=st.sidebar.button("Visualize Data ")
if graphs==True:
  df_image=Image.open('D _ F.png')
  st.image(df_image, caption='DISTANCE VS FARE AMOUNT')
  st.write("")
  pf_image=Image.open('P _F.png')
  st.image(pf_image, caption='PASSENGER COUNT VS FARE AMOUNT')
  st.write("")
  tf_image=Image.open('T_F.png')
  st.image(tf_image, caption='TIME VS FARE AMOUNT')
  comp_image=Image.open('Compare.png')
  st.image(comp_image, caption='COMPARISON')

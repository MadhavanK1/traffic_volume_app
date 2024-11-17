# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

st.title('Traffic Volume Predictor')
st.write("Utilize our advanced Machine Learning application to predict traffic volume.")
st.image('traffic_image.gif')

model_pickle = open('xgboost_model.pickle', 'rb') 
xgb_model = pickle.load(model_pickle) 
model_pickle.close()

mapie_pickle = open('mapie_model.pickle', 'rb') 
mapie_model = pickle.load(mapie_pickle) 
mapie_pickle.close()


sample_df = pd.DataFrame({
    'holiday': ['None', 'None', 'None', 'None', 'None'],
    'temp': [288.28, 289.36, 289.58, 290.13, 291.14],
    'rain_1h': [0.0, 0.0, 0.0, 0.0, 0.0],
    'snow_1h': [0.0, 0.0, 0.0, 0.0, 0.0],
    'clouds_all': [40, 75, 90, 90, 75],
    'weather_main': ['Clouds', 'Clouds', 'Clouds', 'Clouds', 'Clouds'],
    'month': ['October', 'October', 'October', 'October', 'October'],
    'weekday': ['Tuesday', 'Tuesday', 'Tuesday', 'Tuesday', 'Tuesday'],
    'hour': [9, 10, 11, 12, 13]
})

st.sidebar.image('traffic_sidebar.jpg', caption = "Traffic Volume Predictor")
st.sidebar.write("## Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

with st.sidebar.expander("Option 1: Upload CSV File"):
    uploaded_file = st.file_uploader("Upload a CSV file containing the diamond details.", type=['csv'])
    st.dataframe(sample_df)
    st.warning('⚠️ Ensure your uploaded file has the same column names and data types as shown above.')

with st.sidebar.expander("Option 2: Fill Out Form"):
    with st.form("user_inputs_form"):
        st.header("Enter traffic prediction details")
        holiday = st.selectbox('Choose whether today is a designated holiday or not', options=['None', 'Columbus Day', 'Veterans Day', 
                                                 'Thanksgiving Day', 'Christmas Day', 
                                                 'New Years Day', 'Independence Day', 
                                                 'State Fair', 'Labor Day', 'Memorial Day'])
        temp = st.number_input('Average temperature in Kelvin', min_value=250.0, max_value=320.0, value=288.28)
        rain_1h = st.number_input('Amount in mm of rain that occurred in the hour', min_value=0.0, max_value=100.0, value=0.0)
        snow_1h = st.number_input('Amount in mm of snow that occurred in the hour', min_value=0.0, max_value=100.0, value=0.0)
        clouds_all = st.number_input('Percentage of cloud cover', min_value=0, max_value=100, value=40)
        weather_main = st.selectbox('Choose the current weather', options=['Clear', 'Clouds', 'Rain', 
                                                                'Snow', 'Mist', 'Drizzle', 
                                                                'Haze', 'Thunderstorm', 'Fog'])
        month = st.selectbox('Choose month', options=['January', 'February', 'March', 'April', 
                                             'May', 'June', 'July', 'August', 
                                             'September', 'October', 'November', 'December'])
        weekday = st.selectbox('Choose day of the week', options=['Monday', 'Tuesday', 'Wednesday', 
                                                  'Thursday', 'Friday', 'Saturday', 'Sunday'])
        hour = st.number_input('Choose hour', min_value=0, max_value=23, value=9)
        submit_button = st.form_submit_button("Predict Traffic Volume")

if uploaded_file is None and not submit_button:
    st.info('📤 Please choose a data input method to proceed.')
elif uploaded_file:
    st.success('✅ CSV file uploaded successfully.')
elif submit_button:
    st.success('✅ Form data submitted successfully.')

alpha = st.slider('Select alpha value for prediction intervals', 0.01, 0.99, 0.10, 0.01)

def encode_input(data):
    encode_df = data.copy()
    
    categorical_columns = ['holiday', 'weather_main', 'month', 'weekday']
    encoded_df = pd.get_dummies(encode_df, columns = categorical_columns)

    expected_columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour',
                       'holiday_Columbus Day', 'holiday_Independence Day', 
                       'holiday_Labor Day', 'holiday_Martin Luther King Jr Day',
                       'holiday_Memorial Day', 'holiday_New Years Day', 
                       'holiday_State Fair', 'holiday_Thanksgiving Day',
                       'holiday_Veterans Day', 'holiday_Washingtons Birthday',
                       'weather_main_Clouds', 'weather_main_Drizzle',
                       'weather_main_Fog', 'weather_main_Haze',
                       'weather_main_Mist', 'weather_main_Rain',
                       'weather_main_Smoke', 'weather_main_Snow',
                       'weather_main_Squall', 'weather_main_Thunderstorm',
                       'month_August', 'month_December', 'month_February',
                       'month_January', 'month_July', 'month_June',
                       'month_March', 'month_May', 'month_November',
                       'month_October', 'month_September',
                       'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
                       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday']
    
    for col in expected_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
            
    return encoded_df[expected_columns]

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    encoded_df = encode_input(input_df)
    predictions, intervals = mapie_model.predict(encoded_df, alpha=alpha)

    lower_bounds = np.maximum(intervals[:, 0], 0)
    
    input_df['Predicted Volume'] = predictions.round(0).astype(int)
    input_df['Lower Prediction Limit'] = lower_bounds.round(0).astype(int)
    input_df['Upper Prediction Limit'] = intervals[:, 1].round(0).astype(int)

    prediction_interval = int(((1-alpha)*100))
    
    st.write(f"### Prediction Results with {prediction_interval}% Prediction Interval:")
    st.dataframe(input_df)

if submit_button:
    user_data = pd.DataFrame({
        'holiday': [holiday],
        'temp': [temp],
        'rain_1h': [rain_1h],
        'snow_1h': [snow_1h],
        'clouds_all': [clouds_all],
        'weather_main': [weather_main],
        'month': [month],
        'weekday': [weekday],
        'hour': [hour]
    })
    
    encoded_df = encode_input(user_data)
    prediction, intervals = mapie_model.predict(encoded_df, alpha=alpha)

    pred_value = prediction[0]
    lower_limit = min(intervals[0][0], intervals[0][1])
    upper_limit = max(intervals[0][0], intervals[0][1])
    
    st.write("## Predicting Traffic Volume...")
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value.item():,.0f}")
    st.write(f"**Prediction Interval ({(1-alpha)*100:.0f}%):** [{lower_limit.item():,.0f}, {upper_limit.item():,.0f}]")

st.subheader("Model Performance and Inference")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('traffic_feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('traffic_residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Predicted Vs. Actual")
    st.image('traffic_pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('traffic_coverage.svg')
    st.caption("Range of predictions with confidence intervals.")





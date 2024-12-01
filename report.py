import pandas as pd
import streamlit as st
# from sklearn.neighbors import KNeighborsClassifier

import datetime

import joblib

def load_data(data) -> pd.DataFrame:
    return pd.read_csv(data)

df = load_data("data/american_bankruptcy.csv")

# Load model
model = joblib.load("models/model_knn_tune.pkl")

# Side bar

option = st.sidebar.selectbox(
    "Menu",
    ("Predict")
)

if 'Predict' in option: 

    st.logo("data:image/jpeg;base64,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")

    st.title(":red[Temasek POLYTECHNIC]")
    st.title("Cloud Computing & Machine Learning (CCML) Project Proposal")

    st.markdown('''
    :blue[Small cap (< 1 billion) company bankruptcy prediction]''')

    with st.expander("See explanation"):
        st.write('''
        This prediction model uses 19 features to predict whether the company is alive or failed.
        ''')
        st.image("https://images.unsplash.com/photo-1455849318743-b2233052fcff?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fHN0YXJ0dXB8ZW58MHx8MHx8fDA%3D")

    def user_input_features():

        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input("Year", value=2000, min_value=1999, max_value = int(datetime.date.today().strftime("%Y")), placeholder="Type the year")
            current_assets = st.number_input("Current Asset", placeholder="Type a number")
            cost_of_goods_sold = st.number_input("Cost of Goods Sold", placeholder="Type a number")
            depreciation_amortization = st.number_input("Depreciation and Amortization", placeholder="Type a number")
            ebitda = st.number_input("EBITDA", placeholder="Type a number")
            inventory = st.number_input("Inventory", placeholder="Type a number")
            net_income = st.number_input("Net Income", placeholder="Type a number")

    
        with col2:
            total_receivables = st.number_input("Total Receivables", placeholder="Type a number")
            market_value = st.number_input("Market Value (in miilions)",  placeholder="Type a number")
            net_sales = st.number_input("Net Sales", placeholder="Type a number")
            total_assets = st.number_input("Total Assets", placeholder="Type a number")
            total_long_term_debt = st.number_input("Total Long Term Debt", placeholder="Type a number")
            ebit = st.number_input("EBIT",  placeholder="Type a number")
        
        with col3:
            gross_profit = st.number_input("Gross Profit",  placeholder="Type a number")
            total_current_liabilities = st.number_input("Total Current Liabilities",  placeholder="Type a number")
            retained_earnings = st.number_input("Retained Earnings",  placeholder="Type a number")
            total_revenue = st.number_input("Total Revenue",  placeholder="Type a number")
            total_liabilities = st.number_input("Total Liabilies",  placeholder="Type a number")
            total_operating_expenses = st.number_input("Total Operating Expenses",  placeholder="Type a number")
    

        data = {'year': year,
            'Current assets': current_assets,
            'Cost of goods sold': cost_of_goods_sold,
            'Depreciation and amortization': depreciation_amortization,
            'EBITDA': ebitda,
            'Inventory': inventory,
            'Net Income': net_income,
            'Total Receivables': total_receivables,
            'Market value': market_value,
            'Net sales': net_sales,
            'Total assets': total_assets,
            'Total Long-term debt': total_long_term_debt,
            'EBIT': ebit,
            'Gross Profit': gross_profit,
            'Total Current Liabilities':total_current_liabilities,
            'Retained Earnings': retained_earnings,
            'Total Revenue': total_revenue,
            'Total Liabilities': total_liabilities,
            'Total Operating Expenses': total_operating_expenses
           }
        features = pd.DataFrame(data, index=[0])
        return features

    df_feature = user_input_features()

    if st.button("Submit data"):    
        # Make predictions using the loaded model with new data
        predicted_status = model.predict(df_feature)
    
        status = predicted_status[0]
     
        # Predict using the model
        st.divider()    
        st.subheader('Prediction')
        if predicted_status[0] == "alive":
        #    st.markdown('''Predicted status: :green[status]''')
            st.write("Predicted status:", predicted_status[0])
            st.image("https://plus.unsplash.com/premium_photo-1661753066665-dc14526fc1be?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8c21hbGwlMjBtZWRpdW0lMjBlbnRlcnByaXNlfGVufDB8fDB8fHww")
        else:
            st.write("Predicted status:", predicted_status[0])
            st.image("https://images.unsplash.com/photo-1556327070-9661a89d51db?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fGJ1c2luZXNzJTIwY2xvc2VkfGVufDB8fDB8fHww")

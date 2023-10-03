OPENAI_API_KEY=""

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')
df.head(3)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.to_csv('titanic_age_fillna.csv', index=False)

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    ["titanic.csv", "titanic_age_fillna.csv"],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
#agent.run("how many rows in the age column are different between the two dfs?")
#agent.run("what rows is different between the two dfs?")
agent.run("what is the total of the fare column in the two dfs?")
#agent.run("how many rows are there?")

st.write(
    '''
    # Stock Tracker
    '''
)
stock_selection = st.selectbox('Please choose the stock you want to know about ',
                               ['GOOGL', 'MSFT', 'AMZN', 'TSLA', 'AAPL'])

tickerSymbol = stock_selection
tickerData = yf.Ticker(tickerSymbol)
tickerDF = tickerData.history(
    period='max')
st.line_chart(tickerDF.Close)

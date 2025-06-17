import warnings
from crewai import Agent,Task,Crew
import os
from crewai_tools import  SerperDevTool,ScrapeWebsiteTool
from crewai.tools import BaseTool
from pydantic import BaseModel
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

api_key=""
#os.environ["MODEL_NAME"]=""
#os.environ["SERPER_API_KEY"] = get_serper_api_key()

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

data_analyst_agent = Agent(
role='Data Analyst',
goal="Monitor and analyze market data in real-time to identify trei",
backstory="Specializing in financial markets, this agent uses stat:",
verbose =True,
allow_delegation=True,
tools = [scrape_tool, search_tool]
)

trading_strategy_agent = Agent (
role="Trading Strategy Developer",
goal="Develop and test various trading strategies based on insight:",
backstory="Equipped with a deep understanding of financial markets",
verbose=True,
allow_delegation=True,
tools =[scrape_tool, search_tool]
)

execution_agent= Agent(
role='Trade Advisor',
goal="Suggest optimal trade execution strategies based on approved",
backstory="This agent specializes in analyzing the timing, price,",
verbose =True,
allow_delegation=True,
tools = [scrape_tool, search_tool]
)

risk_management_agent = Agent(
role='Risk Avisor',
goal='Evaluate and provide insights on the risks associated with pe',
backstory='Armed with a deep understanding of risk assessment mode',
verbose =True,
allow_delegation=True,
tools = [scrape_tool,search_tool]
)

#Task for Data Analyst Agent: Analyze Market Data

data_analysis_task = Task(

description=(
"Continuously monitor and analyze market data for the selected "
"Use statistical modeling and machine learning to identify tre ),"),
expected_output=(
"Insights and alerts about significant market opportunities or ),"
),
agent=data_analyst_agent,
)

#Task for Trading Strategy Agent: Develop Trading Strategies

strategy_development_task =Task(
description=(
"Develop and refine trading strategies based on the insights fi"
"user-defined risk tolerance ({risk_tolerance)). Consider trad ),"
),
expected_output=(
"A set of potential trading strategies for {stock_selection) ti ),"
),
agent=trading_strategy_agent
)



execution_planning_task = Task(
description=(
"Analyze approved trading strategies to determine the best exe ""considering current market conditions and optimal pricing."),
expected_output=(
"Detailed execution plans suggesting how and when to execute t 1."),
agent=execution_agent,
)

risk_assessment_task = Task(
description=(
"Evaluate the risks associated with the proposed trading strate ""Provide a detailed analysis of potential risks and suggest mi"),
expected_output=(
"A comprehensive risk analysis report detailing potential risks "),
agent=risk_management_agent,
)



# Define the crew with agents and tasks

financial_trading_crew =Crew(
agents=[data_analyst_agent, trading_strategy_agent, execution_agent],
tasks=[data_analysis_task, strategy_development_task, execution_planning_task],
manager_llm=ChatOpenAI(model="gpt-4-turbo", temperature =0.7),
process= Process.hierarchical,
verbose =True
)

#Example data for kicking off the process

finanial_trading_inputs = {

'stock selection': 'AAPL',

'initial capital': '100000',

'risk tolerance': 'Medium',

'trading_strategy_preference': 'Day Trading',

'news_impact_consideration': True
}

result = financial_trading_crew.kickoff(inputs=finanial_trading_inputs)
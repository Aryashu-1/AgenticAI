from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import initialize_agent
import os 

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

result = llm.invoke("what is ipl")
print(result)
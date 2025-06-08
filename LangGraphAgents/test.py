from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-v2"
)

response = llm.invoke([{"role": "user", "content": "Hello"}])
print(response.content)

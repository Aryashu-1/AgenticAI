from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langgraph.graph import add_messages,StateGraph,END
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()

sqlite_con = sqlite3.connect("checkpoint.sqlite",check_same_thread=False)

llm = ChatGroq(
    
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

memory = SqliteSaver(sqlite_con)


class BasicChatState(TypedDict) :
  messages:Annotated[list,add_messages]

def chatbot(state:BasicChatState):
  return {
    "messages": [llm.invoke(state["messages"])] 
  }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)


app = graph.compile(checkpointer=memory)

config = {
  "configurable":{
    "thread_id":1
  }
}



while True:
  user_input = input("You: ")
  if user_input in ["exit","end","close","bye"]:
    break
  else:
    result = app.invoke({
      "messages": [HumanMessage(content = user_input)]
    },config=config)
    
    print("Bot: ",result["messages"][-1].content)


from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langgraph.graph import add_messages,StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv()

llm = ChatGroq(
    
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

tool = TavilySearchResults(max_results=1)

tools =[tool]

llm_with_tools= llm.bind_tools(tools)

class BasicChatState(TypedDict) :
  messages:Annotated[list,add_messages]

def chatbot(state:BasicChatState):
  return {
    "messages": [llm_with_tools.invoke(state["messages"])] 
  }

def tools_router(state:BasicChatState):
  last_message = state["messages"][-1]

  if (hasattr(last_message,"tool_calls") and len(last_message.tool_calls )>0):
    return "tools"
  else :
    return END

tool_node = ToolNode(tools=tools)
graph = StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_node("tools",tool_node)
graph.add_conditional_edges("chatbot",tools_router)
graph.add_edge("tools","chatbot")



app = graph.compile(checkpointer=memory,interrupt_before=["tools"])

app.get_graph().draw_mermaid_png()

config = { "configurable":{
  "thread_id":1,
}
}

# while True:
  # user_input = input("You: ")
  # if user_input in ["exit","end","close","bye"]:
  #   break
  # else:
result = app.stream({
      "messages": [HumanMessage(content = "what is weather in hyderabad")]
    },config=config,stream_mode="values")
    
for event in result:
  event["messages"][-1].pretty_print()




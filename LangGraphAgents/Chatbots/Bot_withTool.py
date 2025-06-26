from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langgraph.graph import add_messages,StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
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
    return "tool_node"
  else :
    return END

tool_node = ToolNode(tools=tools)
graph = StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_node("tools_node",tool_node)
graph.add_conditional_edges("chatbot",tools_router)
graph.add_edge("chatbot","tools_node")
graph.add_edge("chatbot",END)


app = graph.compile()

app.get_graph().draw_mermaid_png()

while True:
  user_input = input("You: ")
  if user_input in ["exit","end","close","bye"]:
    break
  else:
    result = app.invoke({
      "messages": [HumanMessage(content = user_input)]
    })
    
    print("Bot: ",result["messages"][-1].content)




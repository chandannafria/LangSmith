from langchain_ollama import ChatOllama
from langchain_core.messages import (HumanMessage,SystemMessage,ToolMessage)
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import wikipedia


# =========================================================
# 1. LLM
# =========================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# =========================================================
# 2. DuckDuckGo Search
# =========================================================

search = DuckDuckGoSearchRun()


@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Use this for current or general web information.
    """
    try:
        return search.run(query)
    except Exception as e:
        return f"DuckDuckGo search failed: {e}"


# =========================================================
# 3. Wikipedia Search
# =========================================================

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual information about a topic.
    """
    try:
        return wikipedia.summary(
            query,
            sentences=3
        )

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple Wikipedia results found: {e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return "Wikipedia page not found."

    except Exception as e:
        return f"Wikipedia search failed: {e}"


# =========================================================
# 4. Create Tool List
# =========================================================

tools = [
    duckduckgo_search,
    wikipedia_search
]


# =========================================================
# 5. Bind Tools to LLM
# =========================================================

llm_with_tools = llm.bind_tools(tools)


# =========================================================
# 6. Tool Mapping
# =========================================================

tool_map = {
    "duckduckgo_search": duckduckgo_search,
    "wikipedia_search": wikipedia_search
}


# =========================================================
# 7. System Message
# =========================================================

system_message = SystemMessage(
    content="""
You are a helpful AI research assistant.

Use Wikipedia when the user asks for general
factual or historical information.

Use DuckDuckGo when the user asks for current,
recent, or web-based information.

If no tool is required, answer directly.
"""
)


# =========================================================
# 8. Function to Run Agent
# =========================================================

def run_agent(question: str):

    messages = [
        system_message,
        HumanMessage(content=question)
    ]

    # -----------------------------------------------------
    # First LLM call
    # -----------------------------------------------------

    response = llm_with_tools.invoke(messages)

    # Add AI response to conversation
    messages.append(response)

    # -----------------------------------------------------
    # Tool execution loop
    # -----------------------------------------------------

    while response.tool_calls:

        tool_messages = []

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print("\n==============================")
            print("TOOL CALLED:", tool_name)
            print("ARGUMENTS:", tool_args)
            print("==============================")

            # Find tool
            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:

                result = f"Tool '{tool_name}' not found."

            else:

                # Execute tool
                result = selected_tool.invoke(tool_args)

            print("\nTOOL RESULT:")
            print(result)

            # Create ToolMessage
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id
            )

            tool_messages.append(tool_message)

        # Add tool results to conversation
        messages.extend(tool_messages)

        # -------------------------------------------------
        # Send tool result back to LLM
        # -------------------------------------------------

        response = llm_with_tools.invoke(messages)

        messages.append(response)

    # =====================================================
    # Final Answer
    # =====================================================

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")

    print(response.content)


# =========================================================
# 9. Run Application
# =========================================================

if __name__ == "__main__":

    question = input("\nAsk your question: ")

    run_agent(question)
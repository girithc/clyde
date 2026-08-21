from dotenv import load_dotenv

load_dotenv()  # Load FIREWORKS_API_KEY etc. before agents build the LLM

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from agents import default_graph as graph, default_system_prompt as system_prompt


# --- Terminal REPL Interface ---

def main():
    print("🤖 Agent initialized! Type 'exit' to quit.\n")

    # Store chat history
    conversation_history = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_input = input("User > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            conversation_history.append(HumanMessage(content=user_input))

            # Execute agent graph step-by-step
            for event in graph.stream({"messages": conversation_history}, stream_mode="values"):
                latest_msg = event["messages"][-1]

                # Show tool calls
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    for call in latest_msg.tool_calls:
                        print(f"🔧 [Tool Call]: {call['name']} -> Args: {call['args']}")

                # Show tool responses
                elif isinstance(latest_msg, ToolMessage):
                    print(f"📥 [Tool Output]:\n{latest_msg.content.strip()}\n")

                # Show final text response
                elif latest_msg.content and latest_msg != conversation_history[-1]:
                    print(f"\nAgent > {latest_msg.content}\n")

            # Update master history from graph state
            conversation_history = event["messages"]

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

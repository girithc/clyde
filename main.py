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
            # Render every new message. Parallel tool calls append several
            # ToolMessages at once, so don't look only at messages[-1] — that
            # silently drops all but the last result.
            seen = len(conversation_history)
            for event in graph.stream(
                {"messages": conversation_history},
                stream_mode="values",
                config={"recursion_limit": 100},
            ):
                for latest_msg in event["messages"][seen:]:
                    # Show tool calls
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        for call in latest_msg.tool_calls:
                            print(f"🔧 [Tool Call]: {call['name']} -> Args: {call['args']}")

                    # Show tool responses
                    elif isinstance(latest_msg, ToolMessage):
                        print(f"📥 [Tool Output]:\n{latest_msg.content.strip()}\n")

                    # Show final text response
                    elif latest_msg.content:
                        print(f"\nAgent > {latest_msg.content}\n")
                seen = len(event["messages"])

            # Update master history from graph state
            conversation_history = event["messages"]

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

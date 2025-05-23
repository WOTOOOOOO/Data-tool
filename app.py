# Third-party imports
import streamlit as st

from agent import SalesDataAgent


class SalesDataChatApp:
    def __init__(self, csv_path: str):
        """Initialize the chat app with the SalesDataAgent."""
        self._initialize_session_state(csv_path)

    def _initialize_session_state(self, csv_path):
        """Ensure session state variables exist."""
        if "agent" not in st.session_state:
            st.session_state.agent = SalesDataAgent(csv_path)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def run(self):
        """Run the Streamlit app."""
        st.title("📊 Sales Data Analysis Agent")

        # User input
        user_input = st.chat_input("Ask a question about the sales data:")
        if user_input:
            # Store user query
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get response from agent
            response = st.session_state.agent.query(user_input)

            # Store assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(f"<p style='font-size:16px;'>{chat["content"]}</p>", unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    chat_app = SalesDataChatApp("sales_data.csv")
    chat_app.run()

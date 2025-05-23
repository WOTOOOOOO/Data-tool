# Standard library imports
from dotenv import load_dotenv

# Third-party imports
import pandas as pd

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain.chains import SequentialChain, TransformChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType



class SalesDataAgent:
    def __init__(self, csv_path: str):
        """
        Initializes the SalesDataAgent with raw sales data and sets up processing tools.

        Args:
            csv_path (str): Path to the CSV file containing sales data.
        """
        load_dotenv()
        """Initialize the SalesDataAgent with raw data and processing pipeline."""
        self.df_raw = pd.read_csv(csv_path)
        self.llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

        # Define memory for conversation tracking
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define processing chains
        self.processed_results = self._run_data_processing()

        # Initialize Pandas agents
        self.agent_raw = create_pandas_dataframe_agent(self.llm, self.df_raw, verbose=True, allow_dangerous_code=True)

        # Set up tools
        raw_data_tool = Tool(
            name="Raw pandas Data frame querying tool",
            func=self.agent_raw.run,
            description="Provides insights and queries on the raw sales data."
        )
        product_sales_tool = Tool(
            name="Product Sales Statistics Analysis Tool",
            func=self.query_product_sales,
            description="Provides revenue and quantity details for products."
        )

        payment_analysis_tool = Tool(
            name="Payment Method Statistics Analysis Tool",
            func=self.query_payment_analysis,
            description="Provides insights on payment method distribution and average order value."
        )

        # Initialize conversational agent
        self.meta_agent = initialize_agent(
            tools=[raw_data_tool, product_sales_tool, payment_analysis_tool],
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def _clean_and_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the sales data for analysis.

        Args:
            data (pd.DataFrame): Raw sales data.

        Returns:
            pd.DataFrame: Preprocessed sales data with standardized column names,
                          converted data types, and duplicate removal.
        """
        data = data.copy()

        # Standardizing column names
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

        # Convert necessary columns to appropriate data types
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)
        data["price"] = pd.to_numeric(data["price"], errors="coerce").fillna(0)
        data["total_price"] = pd.to_numeric(data["total_price"], errors="coerce").fillna(0)
        data["discount_applied"] = pd.to_numeric(data["discount_applied"], errors="coerce").fillna(0)
        data["final_price"] = pd.to_numeric(data["final_price"], errors="coerce").fillna(0)

        # Fill missing categorical values with "Unknown"
        categorical_cols = ["product", "customer_id", "customer_name", "region", "payment_method"]
        data[categorical_cols] = data[categorical_cols].fillna("Unknown")

        # Drop duplicates
        data = data.drop_duplicates()

        return data

    def _analyze_product_sales(self, data: pd.DataFrame) -> dict:
        """
        Analyzes product sales trends based on revenue and quantity sold.

        Args:
            data (pd.DataFrame): Preprocessed sales data.

        Returns:
            dict: Dictionary mapping products to total revenue and quantity sold.
        """
        product_sales = (
            data.groupby("product")
            .agg({"final_price": "sum", "quantity": "sum"})
            .sort_values(by=["final_price"], ascending=False)
            .to_dict(orient="index")
        )

        return product_sales

    def _evaluate_payment_methods(self, data: pd.DataFrame) -> dict:
        """
        Evaluates payment method trends, including distribution and average order value.

        Args:
            data (pd.DataFrame): Preprocessed sales data.

        Returns:
            dict: Dictionary with payment method distribution and average order values.
        """
        payment_distribution = data["payment_method"].value_counts(normalize=True).to_dict()
        avg_order_value = data.groupby("payment_method")["final_price"].mean().to_dict()

        return {
            "distribution": payment_distribution,
            "avg_order_value": avg_order_value,
        }

    def _run_data_processing(self) -> dict:
        """
        Runs a multi-step processing pipeline using TransformChain.

        Returns:
            dict: Processed sales data including cleaned data, product sales analysis,
                  and payment method insights.
        """
        cleaning_chain = TransformChain(
            input_variables=["data"],
            output_variables=["cleaned_data"],
            transform=lambda inputs: {"cleaned_data": self._clean_and_preprocess(pd.read_json(inputs["data"]))}
        )

        product_sales_chain = TransformChain(
            input_variables=["cleaned_data"],
            output_variables=["product_sales"],
            transform=lambda inputs: {
                "product_sales": self._analyze_product_sales(pd.DataFrame(inputs["cleaned_data"]))}
        )

        payment_analysis_chain = TransformChain(
            input_variables=["cleaned_data"],
            output_variables=["payment_analysis"],
            transform=lambda inputs: {
                "payment_analysis": self._evaluate_payment_methods(pd.DataFrame(inputs["cleaned_data"]))}
        )

        multi_step_chain = SequentialChain(
            chains=[cleaning_chain, product_sales_chain, payment_analysis_chain],
            input_variables=["data"],
            output_variables=["cleaned_data", "product_sales", "payment_analysis"],
            verbose=True
        )

        # Run the chain on raw data
        results = multi_step_chain.invoke({"data": self.df_raw.to_json()})

        return results

    def query_product_sales(self, query) -> dict:
        """
        Retrieves product sales statistics.

        Args:
            query: Placeholder argument (not used in current implementation).

        Returns:
            dict: Product sales analysis results.
        """
        return self.processed_results["product_sales"]

    def query_payment_analysis(self, query) -> dict:
        """
        Retrieves payment method statistics.

        Args:
            query: Placeholder argument (not used in current implementation).

        Returns:
            dict: Payment method analysis results.
        """
        return self.processed_results["payment_analysis"]

    def query(self, user_query: str) -> str:
        """
        Handles user queries by passing them to the conversational agent.

        Args:
            user_query (str): User's query regarding sales data.

        Returns:
            str: The conversational agent's response.
        """
        return self.meta_agent.run(user_query)

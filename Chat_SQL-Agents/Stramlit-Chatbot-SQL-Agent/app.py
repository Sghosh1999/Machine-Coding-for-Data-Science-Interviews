import streamlit as st
import pandas as pd
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
import matplotlib.pyplot as plt
from streamlit_chat import message
import seaborn as sns
import sqlite3
import os
import json
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser

os.environ["GROQ_API_KEY"] = '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'
llm = ChatGroq(model="llama3-8b-8192")

# Initialize the SQLite database connection
db = SQLDatabase.from_uri("sqlite:///chatbot.db")
engine = sqlite3.connect('chatbot.db')

# Define table structure
table_info = """
    Table: chat_data
    Columns:
    - work_year: INTEGER
    - experience_level: TEXT
    - employment_type: TEXT
    - job_title: TEXT
    - salary: INTEGER
    - salary_currency: TEXT
    - salary_in_usd: INTEGER
    - employee_residence: TEXT
    - remote_ratio: INTEGER
    - company_location: TEXT
    - company_size: TEXT
    """

class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: str = Field(..., description="ONLY provide the SQL Query")


def query_validator_agent(sql_query, table_info):
    base_prompt_template = """
    You are a SQL query assistant. You will be provided with the table schema and a SQL Query. 
    Your task is to check if the SQL query is valid based on the schema and fix any issues that arise, 
    ensuring the query is compatible with SQLite.

    <CRUCIAL RULES>
    - Always provide a valid SQL query based on the schema provided. No additional text is allowed.
    - The SQL Query MUST BE COMPATIBLE WITH SQLITE. SQLite does not support advanced functions such as PERCENTILE_CONT.
    - If the query uses unsupported functions or features, replace them with SQLite-compatible alternatives.
    - Verify the columns and tables in the query.
    - If the query includes percentile calculations, use a workaround or remove them, as SQLite does not support `PERCENTILE_CONT`.

    Table Schema: 
    {SCHEMA}
    
    SQL QUERY:
    {SQL_QUERY}
    
    Correct SQL Query: ONLY provide the SQL query. No additional text is allowed.
    \n{format_instructions}\n
    """

    parser = JsonOutputParser(pydantic_object=QueryOutput)
    
    prompt = PromptTemplate(
        template=base_prompt_template,
        input_variables=["SCHEMA", "SQL_QUERY"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser 
    response = chain.invoke({
        "SCHEMA": table_info,
        "SQL_QUERY": sql_query
    })
    
    # Return the fixed query if any changes are made
    return response['query']

# SQL generation function with enhanced prompt template for SQLite
def generate_sql_query(user_question: str, table_info: str, dialect: str = 'sqlite') -> str:
    """Generate SQL query from user question for SQLite."""
    
    # Example scenarios to guide the model
    example_scenarios = """
    Example Scenarios:
    1. Find the total salary for employees in a specific job title (e.g., 'Software Engineer').
       Query: SELECT SUM(salary) FROM chat_data WHERE job_title = 'Software Engineer';

    2. Retrieve the top 5 highest salaries and their job titles.
       Query: SELECT job_title, salary FROM chat_data ORDER BY salary DESC LIMIT 5;

    3. Find the average salary by experience level, sorted by average salary.
       Query: SELECT experience_level, AVG(salary) FROM chat_data GROUP BY experience_level ORDER BY AVG(salary) DESC;

    4. Retrieve the employees with the lowest salary in a specific department (e.g., 'Engineering').
       Query: SELECT job_title, salary FROM chat_data WHERE job_title = 'Engineering' ORDER BY salary ASC LIMIT 1;

    5. Count the number of employees with a remote ratio greater than 50%.
       Query: SELECT COUNT(*) FROM chat_data WHERE remote_ratio > 50;

    6. Find the 10th percentile salary in the dataset (workaround for SQLite as it doesn't support `PERCENTILE_CONT`).
       Query: SELECT salary FROM chat_data ORDER BY salary LIMIT 1 OFFSET (SELECT ROUND(0.1 * COUNT(*) - 1) FROM chat_data);

    7. List employees who work remotely and have a salary greater than 100000.
       Query: SELECT job_title, salary FROM chat_data WHERE remote_ratio = 100 AND salary > 100000;

    8. Retrieve the employee with the highest salary in each job title.
       Query: SELECT job_title, MAX(salary) FROM chat_data GROUP BY job_title;

    9. Get the average salary for each company size (e.g., 'Small', 'Medium', 'Large').
       Query: SELECT company_size, AVG(salary) FROM chat_data GROUP BY company_size;

    10. Find the employees with more than 5 years of experience and their salary in USD.
        Query: SELECT job_title, salary_in_usd FROM chat_data WHERE experience_level = 'Senior' AND work_year > 5;
    """

    prompt_template = PromptTemplate(
        input_variables=["dialect", "table_info", "input", "example_scenarios"],
        template=f"""
        Given an input question, create a syntactically correct {dialect} query to help find the answer. 
        Unless the user specifies in their question a specific number of examples they wish to obtain, 
        always limit your query to at most 5 results. You can order the results by a relevant column to return the most interesting examples in the database.

        **Important Notes for SQLite**:
        - SQLite does not support advanced functions like `PERCENTILE_CONT` or window functions.
        - Ensure that any complex calculations or percentile calculations are replaced with compatible SQLite logic (e.g., subqueries or custom aggregations).
        - You can use `LIMIT` and `OFFSET` for approximations, especially for cases like percentiles.
        - Avoid using non-SQLite syntax or unsupported functions.

        **Use the following table schema**:
        {table_info}

        **Example Scenarios**:
        {example_scenarios}

        **Question**: {user_question}
        """
    )

    prompt = prompt_template.invoke(
        {
            "dialect": dialect,
            "table_info": table_info,
            "input": user_question,
            "example_scenarios": example_scenarios,
        }
    )
    
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    validated_query = query_validator_agent(result.query, table_info)
    return validated_query


# Execute SQL query
def execute_sql_query(query: str, engine) -> pd.DataFrame:
    """Execute the generated SQL query and return the results."""
    return pd.read_sql_query(query, con=engine)

# Generate Plot
def generate_plot(df: pd.DataFrame, plot_flag: bool) -> Optional[plt.Figure]:
    """Generate a plot based on the DataFrame columns if plot_flag is True."""
    if plot_flag and not df.empty:
        plot = None
        num_columns = len(df.columns)
        
        if num_columns == 1:
            plot = df.plot(kind="bar", title="Single Column Plot")
        elif num_columns == 2:
            plot = df.plot(kind="scatter", x=df.columns[0], y=df.columns[1], title="Scatter Plot")
        elif num_columns > 2:
            plot = sns.pairplot(df)  # Create a pairplot for multiple columns

        plt.show()
        return plot
    return None

# Initialize memory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_history = memory.load_memory_variables({})["chat_history"]

# Define the chatbot logic
def chatbot(user_question: str, engine, table_info: str, plot_flag: bool = False, dialect: str = 'sqlite') -> str:
    """Handle user input, generate SQL query, execute it, and maintain chat history."""
    sql_query = generate_sql_query(user_question, table_info)
    query_result = execute_sql_query(sql_query, engine)
    generate_plot(query_result, plot_flag)
    
    result_str = query_result.to_csv(index=False)
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": [sql_query, result_str]})
    
    return result_str, query_result

# Function to convert DataFrame to markdown table
def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown-formatted table."""
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "|---" * len(df.columns) + "|\n"
    for index, row in df.iterrows():
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown


# Streamlit UI with chat_input
st.title("SQL-Based Chatbot")
# Sidebar for displaying the table schema
st.sidebar.header("Table Schema")
st.sidebar.markdown(table_info)

# Initialize session state for chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! Ask me any question related to your database."}]

# Handle user input using chat_input
if prompt := st.chat_input("Ask a question about the data:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get chatbot response
    response, query_result = chatbot(user_question=prompt, engine=engine, table_info=table_info)
    
    # Convert the query result to markdown format for better display
    markdown_table = df_to_markdown(query_result)
    st.session_state.messages.append({"role": "assistant", "content": markdown_table})
    
    # Optionally display the plot
    #plot_flag = st.checkbox("Show plot of query result")
    #generate_plot(query_result, plot_flag)

# Function to save the chat history in a JSON file
import json
def save_chat_history():
    # Ensure the chat history is stored in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Convert messages to JSON format and save to a file
    with open('chat_history.json', 'w') as json_file:
        json.dump(st.session_state.messages, json_file)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])  # Display string content as markdown
        elif isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])  # Display DataFrame as a table

save_chat_history()

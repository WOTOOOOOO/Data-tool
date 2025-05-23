# Data Analysis tool

## Overview

A simple LLM tool that analyzes provided sales information.

## Features

- **Query capabilities**: Gives you the ability to run queries on your CSV data frame, to extract information
- **Statistical analysis**: Provides statistical analysis of the information

## Installation

1. Clone the repository to your local machine:
   ```
   git clone git@github.com:WOTOOOOOO/Data-tool.git
   cd <code directory>

2. Install requirements:
   ```
   pip install -r requirements.txt

3. Add your Groq GROQ_API_KEY to a .env file.

## How to Run

1. Run data_generator.py to generate sales information
2. Open a terminal in the project directory.
3. Run the app using the Streamlit command:
   ```
   streamlit run app.py


## Additional Notes

1. The application generates its own sales data file

## Limitations

The free version of Groq has token limitations so try to request queries that wont result in massive csv outputs
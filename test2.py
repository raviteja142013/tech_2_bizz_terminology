from typing import Annotated, TypedDict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from prompts.prompt import prompt
from pydantic_schema import BusinessNamePrediction,ColumnMetadata

from pathlib import Path

try:
    # Prefer python-dotenv for consistent loading in any context
    from dotenv import load_dotenv, find_dotenv

    # Try to find .env from current cwd; if not found, look near this file
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # Fallback: .env next to this file (works if debugger changes cwd)
        env_path = Path(__file__).resolve().parent / ".env"

    load_dotenv(dotenv_path=env_path, override=False)  # set override=True if you want .env to win
except Exception as e:
    # Optional: log but don’t crash
    print(f"Warning: could not load .env: {e}")

app = FastAPI(title="Code Improvement Agent API", version="1.0.0")

AZURE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_VERSION=os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_DEPLOYMENT=os.getenv('AZURE_OPENAI_DEPLOYMENT')



# Initialize Azure LLM WITHOUT structured output forcing
llm = AzureChatOpenAI(
    azure_endpoint= AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,  # Your current version works fine
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    openai_api_type="azure",
    temperature=0,
)

# print(llm.invoke("Hello, world!"))

# llm_with_structured_output = llm.with_structured_output(BusinessNamePrediction)
parser = PydanticOutputParser(pydantic_object=BusinessNamePrediction)
# @app.post("/generate_business_terminology")
def generate_business_terminology(request: ColumnMetadata):
    try:
        input_data =  request.model_dump_json()
        # input_metadata = input_data.get("metadata", "")

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=input_data),
            # HumanMessage(content = f'The generated output should be in the following JSON format:\n{BusinessNamePrediction.model_json_schema()}')
            HumanMessage(content = f'The generated output should be in the following JSON format:\n{parser.get_format_instructions()}')

        ]

        response = llm.invoke(messages)
        # generated_text = response.generations[0][0].text
        generated_text = response
        # paresed_output = parser.parse(generated_text)
        print("generated_text:", generated_text)
        # return JSONResponse(content={"predicted_business_name": generated_text})
        return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


input_data = {
  "metadata_id": "MD-003",
  "domain": "Customer",
  "business_context": "Customer Profile",
  "system": "CRM",
  "schema": "dim",
  "table": "customer_profile",
  "column": "acct_open_dt",
  "data_type": "DATE",
  "nullable": False,
  "unit": None,
  "lexical_tokens": ["acct", "open", "dt"],
  "expanded_tokens": ["Account", "Open", "Date"],
  "derivation_logic": "date when account was opened in source system",
  "value_encoding": None,
  "sample_value_profile": {
    "min": "1998-01-01",
    "max": "2025-01-01",
    "distinct_count": None,
    "true_ratio": None
  },
  "approved_glossary_terms": ["Account"],
  "prior_approved_examples": [],
  "naming_constraints": {
    "max_words": 5,
    "boolean_suffix": "Indicator",
    "numeric_suffix": "Amount",
    "date_suffix": "Date",
    "allow_parenthetical": False
  }
}


input = ColumnMetadata.model_validate(input_data)
result = generate_business_terminology(input)
print("result:", result)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
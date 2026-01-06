from typing import Annotated, TypedDict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from LLM.llm import llm
from examples.good_examples import good_examples
from examples.bad_examples import bad_examples
import re 
from langchain_core.output_parsers import PydanticOutputParser

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from prompts.prompt import prompt2
from prompts.scoring_prompt import scoring_prompt
from tech_2_bizz_terminology.schemas.pydantic_schema import BusinessNamePrediction,ColumnMetadata



app = FastAPI(title="Code Improvement Agent API", version="1.0.0")



# llm_with_structured_output = llm.with_structured_output(BusinessNamePrediction)
parser = PydanticOutputParser(pydantic_object=BusinessNamePrediction)
# @app.post("/generate_business_terminology")
def generate_business_terminology(request: ColumnMetadata):
    try:
        input_data =  request.model_dump_json()
        # input_metadata = input_data.get("metadata", "")

        messages = [
            SystemMessage(content=prompt2),
            HumanMessage(content=input_data),
            # HumanMessage(content="score the output using the following criteria {scoring_prompt}"),
            # HumanMessage(content = f'The generated output should be in the following JSON format:\n{BusinessNamePrediction.model_json_schema()}')
            HumanMessage(content = f'The generated output should be in the following JSON format:\n{parser.get_format_instructions()}')

        ]

        response = llm.invoke(messages)
        output_content = response.content

        cleaned = re.sub(r"^```json\s*|\s*```$", "", output_content.strip(), flags=re.MULTILINE)
        return cleaned

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



input_data  = bad_examples[1]
input = ColumnMetadata.model_validate(input_data)
result = generate_business_terminology(input)
print("This is the result:", result)



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
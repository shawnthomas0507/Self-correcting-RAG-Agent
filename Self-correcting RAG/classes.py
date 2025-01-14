from typing import List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no' "
    )


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]

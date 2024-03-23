import json
import openai
import sys
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import logging

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.chains.summarize import load_summarize_chain
import textwrap
import pandas as pd
from pathlib import Path as p
from typing import List, Dict, Callable, Optional, Any
import logging

# Set your OpenAI API key
openai.api_key = "OpenAI API key"

# Call LLM instance
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

loader = PyPDFLoader("<PATH TO FILE>")

pages = loader.load_and_split()
print(len(pages))

map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """

map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
                      Write a concise summary of the following text delimited by triple backquotes.
                      Return your response in bullet points which covers the key points of the text.
                      ```{text}```
                      BULLET POINT SUMMARY:
                      """

combine_prompt = PromptTemplate(
    template=combine_prompt_template, input_variables=["text"]
)

map_reduce_chain = load_summarize_chain(
    llm=chat,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    return_intermediate_steps=True,
)

map_reduce_outputs = map_reduce_chain({"input_documents": pages})

print(map_reduce_outputs)

final_mp_data = []
for doc, out in zip(
    map_reduce_outputs["input_documents"], map_reduce_outputs["intermediate_steps"]
):
    output = {}
    output["file_name"] = p(doc.metadata["source"]).stem
    output["file_type"] = p(doc.metadata["source"]).suffix
    output["page_number"] = doc.metadata["page"]
    output["chunks"] = doc.page_content
    output["concise_summary"] = out
    final_mp_data.append(output)

pdf_mp_summary = pd.DataFrame.from_dict(final_mp_data)
pdf_mp_summary = pdf_mp_summary.sort_values(
    by=["file_name", "page_number"]
)  # sorting the dataframe by filename and page_number
pdf_mp_summary.reset_index(inplace=True, drop=True)
pdf_mp_summary.head()

index = 0
print("[Context]")
print(pdf_mp_summary["chunks"].iloc[index])
print("\n\n [Simple Summary]")
print(pdf_mp_summary["concise_summary"].iloc[index])
print("\n\n [Page number]")
print(pdf_mp_summary["page_number"].iloc[index])
print("\n\n [Source: file_name]")
print(pdf_mp_summary["file_name"].iloc[index])

from enum import Enum
from typing import TypedDict

import instructor
import openai
import requests
import ujson
from pydantic import AnyHttpUrl, BaseModel, Field
import streamlit as st
import google.generativeai as genai


oai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


class SearchResult(TypedDict):
    title: str
    link: str
    content: str


def summarize(text: str) -> str:
    """Summarize the text using gemini-1.5-flash (much larger context window)"""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are an expert news writer. You are a PhD (English) researcher. You need to summarize aarticles given to you in as few words possible without missing out on context.",
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
    )
    tokens = model.count_tokens(text)
    if tokens.total_tokens <= 1000:
        return text
    return model.generate_content(text).text


def get_content(url: str) -> str:
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return summarize(resp.text).strip()
    except Exception as e:
        print(f"Error getting content for {url}: {e}")
        return ""


def search_tool(query: str, num_results: int = 3):
    """Tool to search via Google CSE"""
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    cx = st.secrets.get("GOOGLE_CSE_ID", "")
    base_url = "https://www.googleapis.com/customsearch/v1"
    url = f"{base_url}?key={api_key}&cx={cx}&q={query}&num={num_results}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    json = resp.json()
    assert hasattr(json, "items"), "No items in response"
    res: list[SearchResult] = []
    for item in json["items"]:
        res.append(
            {
                "title": item["title"],
                "link": item["link"],
                "content": get_content(item["link"]) or item["snippet"],
            }
        )
    print(ujson.dumps(res, escape_forward_slashes=False, indent=2))
    return res


class SearchQuery(BaseModel):
    query: str


class FactCheckLabel(str, Enum):
    """The fact check label enum"""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    MISLEADING = "misleading"


class GPTFactCheckModel(BaseModel):
    """expected result format from OpenAI for fact checking"""

    label: FactCheckLabel = Field(None, description="The label of the fact check")
    explanation: str = Field(None, description="The explanation of the fact check")
    sources: list[AnyHttpUrl] = Field([], description="The sources of the fact check")


def fact_check_with_gpt(claim: str):
    """
    fact_check_with_gpt checks the data against the OpenAI API.

    Parameters
    ----------
    data : TextInputData
        The data to be checked.

    Returns
    -------
    FactCheckResponse
        The result of the fact check.
    """

    claim = summarize(claim)

    client = instructor.from_openai(oai_client)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=SearchQuery,
        messages=[
            {
                "role": "system",
                "content": "I want you to act as a fact-check researcher. You will be given a claim and you have should search the information on a custom search engine to help in the fact checking. Frame a query using the least words possible and return only the query.",
            },
            {
                "role": "user",
                "content": claim,
            },
        ],
    )
    assert isinstance(response, SearchQuery)

    search_results = search_tool(response.query)

    # Send the search results back to GPT for analysis
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "I want you to act as a fact checker. You will be given a statement along with relevant data and you are supposed to provide a fact check based on them. You need to classify the claim as either 'correct', 'incorrect', or 'misleading' and provide the explanation along with the sources you used.",
            },
            {
                "role": "user",
                "content": f"Original statement: {claim}\n\nSearch results: {ujson.dumps(search_results, escape_forward_slashes=False)}",
            },
        ],
        response_model=GPTFactCheckModel,
    )
    assert isinstance(final_response, GPTFactCheckModel)

    st.json(final_response.model_dump())


st.title("Newsful")
st.subheader("Fact Checker")

claim = st.text_area("Enter a claim to check", height=200)

if st.button("Fact Check"):
    claim = " ".join(claim.split())
    if not claim:
        st.error("Please enter a claim")
        st.stop()
    fact_check_with_gpt(claim)

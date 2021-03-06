from enum import Enum
from typing import Optional, Dict
import config

from fastapi import FastAPI
from pydantic import BaseModel

import openai, chronological
from chronological import read_prompt, append_prompt, cleaned_completion, main
from fastapi.middleware.cors import CORSMiddleware

# Create config.py and add the key as gpt_key for both libraries (or directly enter it here)
openai.api_key = config.gpt_key
chronological.set_api_key(config.gpt_key)




class Languages(str, Enum):
    English = 'en'
    German = 'de'
    Spanish = 'es'


class Tonality(str, Enum):
    Formal = 'formal'
    Friendly = 'friendly'
    Descriptive = 'descriptive'


class Query(BaseModel):
    user_prompt: str
    input_language: Optional[Languages] = 'en'
    output_language: Optional[Languages] = 'en'
    tonality: Optional[Tonality] = 'friendly'
    max_suggestions: Optional[int] = 6


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/findwords/")
async def ask_gpt(query: Query):
    prompt = build_prompt(query)
    # TODO max_tokens needs to be calculated based on required number of suggestions
    result = await cleaned_completion(prompt, max_tokens=200, engine='davinci', temperature=0.5, top_p=1,
                                      frequency_penalty=0.2, stop=['}'])
    suggestions = result.replace('"', '').replace('\n', '').split(';')[:query.max_suggestions]
    suggestions_dict = {sub.split(":")[0].strip(): sub.split(":")[1].strip() for sub in suggestions}

    return {
        "suggestions": suggestions_dict,
        "input_language": query.input_language,
        "output_language": query.output_language,
        "tonality": query.tonality
    }


def build_prompt(user_query: Query):
    if user_query.input_language == Languages.English and \
            user_query.output_language == Languages.English and \
            user_query.tonality == Tonality.Friendly:
        train = read_prompt('en_en_friendly')
    elif user_query.input_language == Languages.German and \
            user_query.output_language == Languages.English and \
            user_query.tonality == Tonality.Friendly:
        train = read_prompt('de_en_friendly')
    else:
        raise NotImplementedError

    return append_prompt(user_query.user_prompt+'}\n\n{', train)

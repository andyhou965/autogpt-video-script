import os
from apikey import API_KEY

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = API_KEY
st.title('My Video Script Creator')

prompt = st.text_input('Type in your Topic here to create a video script')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'], template='write me a video title about {topic}'
)

# With wikipedia research
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} ',
)

# Implement the Memory for APP
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# llm = OpenAI(temperature=0.9)  # type: ignore
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)  # type: ignore

title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key='title',
    memory=title_memory,
)
script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key='script',
    memory=script_memory,
)

# # Can get multiple outputs using SequentialChain
# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain],
#     input_variables=["topic"],
#     output_variables=["title", "script"],
#     verbose=True,
# )

wiki = WikipediaAPIWrapper()  # type: ignore

if prompt:
    # response = sequential_chain({"topic": prompt})
    # title = response["title"]
    # script = response["script"]

    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    # with st.expander('Title History'):
    #     st.info(title_memory.buffer)

    # with st.expander('Script History'):
    #     st.info(script_memory.buffer)

    with st.expander('Wikipedia Research Results'):
        st.info(wiki_research)

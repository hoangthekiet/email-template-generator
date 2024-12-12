import re
from typing import Tuple

import requests
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from readability import Document

from .utils import fix_url

style_learning_prompt = """I will give you an essay. I want you to analyse it in terms of style, focusing specifically on:

- Voice
- Syntax, punctuation, and emoji
- Vocabulary
- Use of anecdotes
- Use of quotes and metaphor
- Pronoun use

Here is the essay for you to analyse:

***
{essay}
***"""

email_writing_prompt = """I will provide you with a topic. Your task is to produce a marketing email of approximately {n} words that closely mimics the style of the provided essay.

The email should be based on factual information, but you may improvise if you are creating anecdotes.

Please maintain the email's sender and receiver names if specified.

Avoid repetitive use of the topic name; employ synonyms or related terms where appropriate.

A marketing email should include the following sections:

- Subject line
- Tag line or slogan (optional)
- Engaging description of the product/service
- Clear and compelling call to action
- Writer's signature

Here is the topic for you to write:

{topic}

**Just directly provide the email content without any additional formatting or explanations.**"""


class WritingAgent():
    def __init__(self):
        # Define the agents
        self.llm_style_learner = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.05)
        self.llm_email_writer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    
        def call_style_learner(state: MessagesState):
            """
            Extract writing style of the given essay.
            """
            messages = state['messages']
            essay = messages[0].essay
            topic = messages[0].topic
            n = messages[0].n
            prompt = style_learning_prompt.format(essay=essay)
            messages[0] = HumanMessage(content=prompt, topic=topic, n=n)
            response = self.llm_style_learner.invoke(messages)
            return {"messages": response}

        def call_email_writer(state: MessagesState):
            """
            Write an email for the given topic using the extracted writing style.
            """
            messages = state['messages']
            topic = messages[0].topic
            n = messages[0].n
            prompt = email_writing_prompt.format(topic=topic, n=n)
            messages.append(HumanMessage(content=prompt))
            response = self.llm_email_writer.invoke(messages)
            return {"messages": response}


        # Define a new graph
        workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent_style_learner", call_style_learner)
        workflow.add_node("agent_email_writer", call_email_writer)

        # Define the edges connecting nodes
        workflow.add_edge(START, "agent_style_learner")
        workflow.add_edge("agent_style_learner", "agent_email_writer")
        workflow.add_edge("agent_email_writer", END)

        # Compile the graph into a LangChain Runnable,
        self.app = workflow.compile()


    def invoke(self, messages: dict, config: dict) -> dict:
        return self.app.invoke(messages,
                               config=config)


    def get_essay(self, url: str) -> Tuple[dict, dict]:
        url = fix_url(url)
        result = requests.get(url).text

        doc = Document(result)
        sum = doc.summary(html_partial=True)

        def _cleanhtml(raw_html):
            CLEANR = re.compile('<.*?>')
            cleantext = re.sub(r"</p>", '\n\n', raw_html)
            cleantext = re.sub(r"<li>", "- ", cleantext)
            cleantext = re.sub(CLEANR, '', cleantext)
            return cleantext.strip()

        return doc.title(), _cleanhtml(sum)

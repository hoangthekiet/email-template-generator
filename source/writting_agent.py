import re
from typing import List, Literal, Tuple

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from readability import Document

from .prompts import (STYLE_LEARNING_PROMPT_0, STYLE_LEARNING_PROMPT_1,
                     EMAIL_WRITING_PROMPT, SYSTEM_PROMPT)
from .utils import resolve_url


class WritingAgent():
    """
    A class representing a multi-agent systems that can analyze an essay's style and generate an email on a given topic.
    The system uses two LLM agents configured in a serial pipeline:
        - The former one for extracting the writing style of an essay
        - The latter one for writing an email in the extracted style
    """

    def __init__(self, verbose: bool = False):
        
        # Define the agents
        """
        Define the WritingAgent with 2 LLM agents and sets up a workflow for processing messages.
        """
        # Style learning agent use very low temperature to ensure consistency
        self.llm_style_learner = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.05)
        # Email writing agent use higher temperature to allow creativity yet avoid hallucinations
        self.llm_email_writer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        # Whether to print verbose messages
        self.verbose = verbose


        def should_continue_analysis(state: MessagesState) -> Literal["agent_style_learner", "agent_email_writer"]:
            """
            Determine which agent should be invoked next based on the presence of essays.

            Returns
            -------
            Returns "agent_style_learner" , otherwise "agent_email_writer".
            """
            system_message = state['messages'][0]
            # If there are essays to analyze, continue learning
            if len(system_message.essays) > 0:
                return "agent_style_learner"
            # Otherwise, break the recursion and switch to the email writer
            return "agent_email_writer"


        def call_style_learner(state: MessagesState):
            """
            Extract writing style of the given essay recursively.
            """
            messages = state['messages']
            # Get essays to use by this agent
            essays = messages[0].essays
            style_learning_prompt = STYLE_LEARNING_PROMPT_0 if len(messages) == 1 else STYLE_LEARNING_PROMPT_1
            prompt = style_learning_prompt.format(essay=essays.pop(0))
            if len(messages) > 1:
                # Only keep the system message to avoid tokens overflow
                messages = messages[:1]
            messages.append(HumanMessage(content=prompt))
            response = self.llm_style_learner.invoke(messages)
            if self.verbose:
                print(f"--\n{response.content}")
            return {"messages": response}


        def call_email_writer(state: MessagesState):
            """
            Write an email for the given topic using the extracted writing style.
            """
            messages = state['messages']
            # Get topic and number of words to use by this agent
            topic = messages[0].topic
            n = messages[0].n
            prompt = EMAIL_WRITING_PROMPT.format(topic=topic, n=n)
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
        workflow.add_conditional_edges("agent_style_learner", should_continue_analysis)
        workflow.add_edge("agent_email_writer", END)

        # Compile the graph into a LangChain Runnable
        self.app = workflow.compile()


    def run(self, essays: List[str], topic: str, num_words: int, config: dict) -> dict:
        """
        Run the multi-agent pipeline with the given messages and configuration.
        """
        return self.app.invoke(input={"messages": [SystemMessage(SYSTEM_PROMPT,
                                                                    essays=essays,
                                                                    topic=topic,
                                                                    n=num_words)]},
                               config=config)


    def get_essay(self, url: str) -> Tuple[str, str]:
        """
        Extract the title and essay from a given URL.

        Parameters
        ----------
        url: The URL of the essay to extract

        Returns
        -------
        A tuple containing the title of the essay and the essay content itself
        """
        url = resolve_url(url)
        result = requests.get(url).text

        doc = Document(result)
        sum = doc.summary(html_partial=True)

        def _cleanhtml(raw_html):
            cleantext = re.sub(r"</p>", '\n\n', raw_html)
            cleantext = re.sub(r"<br>", '\n', cleantext)
            cleantext = re.sub(r"<li>", "- ", cleantext)
            CLEANR = re.compile('<.*?>')
            cleantext = re.sub(CLEANR, '', cleantext)
            return cleantext.strip()

        return doc.title(), _cleanhtml(sum)

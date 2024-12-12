from pprint import pprint

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from source.color_extractor import StyleExtractor
from source.writting_agent import WritingAgent


load_dotenv()

print("What is your website?")
url = input()

style_extractor = StyleExtractor(url)
extracted_style = style_extractor.get_color_profile()
print("--\nExtracted style:")
pprint(extracted_style)

print('\n***\n')

print("Which web pages best demonstrate your writing style?")
url = input()
print("What topic do you want to write about?")
topic = input()
print("How many words should this be? (Input number only)")
num_words = input()

writing_agent = WritingAgent()
title, essay = writing_agent.get_essay(url)
final_state = writing_agent.invoke(messages={"messages": [HumanMessage("",
                                                                       essay=essay,
                                                                       topic=topic,
                                                                       n=num_words)]},
                                   config={"configurable": {"thread_id": 42}})

result = final_state['messages'][-1].content
print('--\nGenerated email:')
print(result)
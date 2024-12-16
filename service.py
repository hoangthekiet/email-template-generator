import os
from pprint import pprint
from tqdm import tqdm

from dotenv import load_dotenv

from src.color_detector import StyleDetector
from src.writting_agent import WritingAgent


load_dotenv()

# Get website for color scheme extraction
print("What is your website?")
url = input()

# Extract color scheme
style_detector = StyleDetector(url)
detected_style = style_detector.get_color_scheme()
print("--\nDetected color scheme:")
pprint(detected_style)

print('\n***\n')

# Get websites for writing style extraction
verbose = os.getenv('VERBOSE', 'false').lower() in ['true', 't', '1']
writing_agent = WritingAgent(verbose=verbose)
print("Which web pages best demonstrate your writing style?\n- Input URLs separated by spaces:", end=' ')
urls = input()
essays = []
for url in tqdm(urls.split()):
    title, essay = writing_agent.get_essay(url)
    essays.append(essay)

# Get topic and number of words
print("What topic do you want to write about?\n- Input content description:", end=' ')
topic = input()
print("How many words should this be?\n- Input number only:", end=' ')
num_words = input()

# Generate content
final_state = writing_agent.run(essays=essays,
                                topic=topic,
                                num_words=num_words,
                                config={"configurable": {"thread_id": 42}})

result = final_state['messages'][-1].object
print('--\nGenerated email:')
pprint(result)

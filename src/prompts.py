SYSTEM_PROMPT = """You are a dedicated email writing assistant.

The user will provide you with a collection of essays to analyze. Based on these, learn the writing style.

Once you have learned it, the user will provide you with a topic. Your task is to compose an email."""


STYLE_LEARNING_PROMPT_0 = """I will give you an essay. I want you to analyze it in terms of style, focusing specifically on:

- Voice
- Syntax, punctuation, and emoji
- Vocabulary
- Use of anecdotes
- Use of quotes and metaphor
- Pronoun use

Here is the essay for you to analyze:

***
{essay}
***"""

STYLE_LEARNING_PROMPT_1 = """I will give you another essay. I want you to refine the previous analysis, continuing to focus on:

- Voice
- Syntax, punctuation, and emoji
- Vocabulary
- Use of anecdotes
- Use of quotes and metaphor
- Pronoun use

Here is the essay for you to analyze:

***
{essay}
***"""

EMAIL_WRITING_PROMPT = """I will provide you with a topic. Your task is to produce a marketing email of approximately {n} words that closely mimics the style of the provided essay.

The email should be based on factual information, but you may improvise if you are creating anecdotes.

Please maintain the email's sender and receiver names if specified.

Avoid repetitive use of the topic name; employ synonyms or related terms where appropriate.

A marketing email should include the following elements:

- Subject line
- Preheader/Snippet
- Tagline/Slogan
- Engaging description of the product/service
- Clear and compelling call to action
- Call to action button of at most 3 words
- Footer
- Sign-offs

Here is the topic for you to write:

{topic}

You will provide the email as a JSON object with the following keys: `subject`, `preheader`, `tagline`, `description`, `call_to_action`, `button`, `footer`, `sign_offs`."""
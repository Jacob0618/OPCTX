"""FROZEN unified prompts (user-specified). Do not modify wording.

Every system (ours and baselines) answers with QA_PROMPT and is scored with
JUDGE_PROMPT. Memory-WRITE prompts are system-specific and live with each system.
"""

QA_PROMPT = """You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.
History Dialogs: {retrieved_texts}
Question: {question}"""

JUDGE_PROMPT = """I will give you a question, a reference answer, and a response from a model. Please answer [[yes]] if the response contains the reference answer. Otherwise, answer [[no]]. If the response is equivalent to the correct answer or contains all the intermediate steps to get the reference answer, you should also answer [[yes]]. If the response only contains a subset of the information required by the answer, answer [[no]].
[User Question]
{question}
[The Start of Reference Answer]
{answer}
[The End of Reference Answer]
[The Start of Model's Response]
{response}
[The End of Model's Response]
Is the model response correct? Answer [[yes]] or [[no]] only."""

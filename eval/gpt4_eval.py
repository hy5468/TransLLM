import requests
import re
import copy
import json
prompt_single = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the thai response provided by an AI assistant to the user thai question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"'''
prompt_single_ref = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the thai response provided by an AI assistant to the user thai question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a english reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]", "description": "Prompt for general questions", "category": "math", "output_format": "[[rating]]"'''
prompt_second = '''Please act as an impartial judge and evaluate the quality of the thai response provided by an AI assistant to the user thai question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n", "prompt_template": "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"'''
prompt_second_ref = '''Please act as an impartial judge and evaluate the quality of the thai response provided by an AI assistant to the user thai question. Your evaluation should consider correctness and helpfulness. You will be given a english reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the  reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n", "prompt_template": "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>", "description": "Prompt for general questions", "category": "math", "output_format": "[[rating]]"'''


input_template = {
    "model":"",
    "messages": [
        {"role": "system", "content": "You are a friendlt and helpful assistant"}
    ],
    "temperature": "0"
}
headers = {"Content-Type": "application/json"}

def get_score(result):
    pattern = r"\[\[(.*?)\]\]"
    score = re.search(pattern=pattern, string=result).group(1)
    return score


# mt_bench_eval
def eval(questions1, answers1, questions2, answers2, refs1, refs2):
    scores1 = []
    scores2 = []
    for i, (question1, question2, answer1, answer2) in enumerate(zip(questions1, questions2,answers1, answers2)):
        if 20<=i<50:
            first_instruct = prompt_single_ref.format_map({"question": question1, "answer": answer1, "ref_answer_1": refs1[i-20]})
        else:
            first_instruct = prompt_single.format_map({"question": question1, "answer": answer1})
        first_input = copy.deepcopy(input_template)
        first_input["messages"].append({"role":"user", "content": first_instruct})
        response1 = requests.post("url", headers=headers, data=json.dumps(first_input)).json()["choices"][0]["messages"]["content"]
        score1 = get_score(response1)

        if 20<=i<50:
            second_instruct = prompt_second_ref.format_map({"question_1": question1, "answer_1": answer1, "question_2": question2, "answer_2": answer2, "ref_answer_1": refs1[i-20], "ref_answer_2": refs2[i-20]})
        else:
            second_instruct = prompt_second.format_map({"question_1": question1, "answer_1": answer1, "question_2": question2, "answer_2": answer2})
        
        second_input = copy.deepcopy(second_instruct)
        second_input["messages"].append(second_instruct)
        response2 = requests.post("url", headers=headers, data=json.dumps(second_input)).json()["choices"][0]["messages"]["content"]
        score2 = get_score(response2)
        scores1.append(score1)
        scores2.append(score2)
    return scores1, scores2

# eval alpaca_eval dataset
"""
def eval(questions, answers):
    scores = []
    for i, (question, answer) in enumerate(zip(questions,answers)):
        first_instruct = prompt_single.format_map({"question": question, "answer": answer})
        first_input = copy.deepcopy(input_template)
        first_input["messages"].append({"role":"user", "content": first_instruct})
        response = requests.post("url", headers=headers, data=json.dumps(first_input)).json()["choices"][0]["messages"]["content"]
        score = get_score(response)
        scores.append(score)
    return scores
"""
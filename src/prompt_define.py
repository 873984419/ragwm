 # WATERMARK_GENERATE = '''
# You are a knowledge graph expert and a linguist. There are two entities (E1 and E2) in one of the knowledge graphs, and the relationship between the two entities is R1. Please restore it to a text based on the relationship R1 between E1 and E2, and ensure the readability of the text. Here is an example: E1: {E1}, E2: {E2}, R1: {R1}; Please restore the text and generate {K} different texts, and pay attention to adjusting the syntactic structure of these {K} sentences to avoid the same. Return only {K} sentences, separated by periods.
# '''
WATERMARK_GENERATE = '''
You are a knowledge graph expert and a linguist. Your task is to generate {K} distinct natural language sentences that describe a given relationship (R1) between two entities (E1 and E2) in a knowledge graph. Each sentence should convey the relationship between (E1) and (E2) using a unique syntactic structure to ensure variation, while maintaining clarity and readability.

Input:
E1: {E1}
E2: {E2}
R1: {R1}

Output:
Return exactly {K} distinct sentences, formatted as a JSON list, like this:
[
    "Sentence 1.",
    "Sentence 2.",
    "Sentence 3.",
    "Sentence 4.",
    ...,
    "Sentence K."
]
'''
WATERMARK_GENERATE_LENGTH  = '''
You are a knowledge graph expert and a linguist. Your task is to generate {K} distinct natural language sentences that describe a given relationship (R1) between two entities (E1 and E2) in a knowledge graph. Each sentence should convey the relationship between (E1) and (E2) using a unique syntactic structure to ensure variation, while maintaining clarity and readability.  Please ensure that the total word count of all sentences combined is approximately (L).
Input:
E1: {E1}
E2: {E2}
R1: {R1}
L:{L}

Output:
Return exactly {K} distinct sentences, each approximately {L} words long, formatted as a JSON list, like this:
[
    "Sentence 1.",
    "Sentence 2.",
    "Sentence 3.",
    "Sentence 4.",
    ...,
    "Sentence K."
]
'''



WATERMARK_GENERATE_FEEDBACK = '''
You are a watermark generator, a knowledge graph expert, and a linguist. In a given knowledge graph, there are two entities (E1 and E2) connected by a relationship (R1). Your task is to generate watermark text (WT) that encodes this relationship (R1) between (E1) and (E2).

This watermark text will be processed by two other language models:

1. **Watermark Extractor**: This model will be asked to identify the relationship (R1) between (E1) and (E2) using the restored watermark text (WT) and additional contextual data. The extractor’s output is labeled as (WE).

2. **Watermark Discriminator**: This model evaluates whether the extractor's output (WE) correctly implies that the relationship (R1) exists between (E1) and (E2). The discriminator’s output is labeled as (WD), with a focus on determining if the relationship is clear and accurate.

Given the following:
- Restored watermark text: ({WT})
- Extractor feedback: ({WE})
- Discriminator feedback: ({WD})
- R1: ({R1})
- E1: ({E1})
- E2: ({E2})

Your goal is to refine the watermark text ({WT}) to ensure that:
1. The relationship (R1) between (E1) and (E2) is preserved after processing by the extractor.
2. The discriminator’s answer (WD) confirms that the relationship (R1) between (E1) and (E2) is still evident and accurate.

Return only the refined watermark text in the following JSON format: 
[{{"watermark_text": "Your generated text 1"}}]
'''

WATERMARK_GENERATE_FEEDBACK_BOTH = '''
You are a watermark generator, a knowledge graph expert, and a linguist. In a given knowledge graph, two entities (E1) and (E2) are connected by a relationship (R1). Your task is to generate watermark text (WT) that clearly encodes this relationship (R1) between (E1) and (E2).

The generated watermark text will undergo two stages of processing:

1. **Direct Evaluation**:
    - **Watermark Discriminator 1 (WD1)**: This model evaluates whether the watermark text (WT) accurately implies the relationship (R1) between (E1) and (E2).

2. **Extractor-Based Evaluation**:
    - **Watermark Extractor (WE)**: This model attempts to extract the relationship (R1) between (E1) and (E2) based on the restored watermark text (WT) and additional contextual data.
    - **Watermark Discriminator 2 (WD2)**: After the extraction, this model assesses whether the relationship (R1) is still clearly and accurately implied.

Your objective is to refine the watermark text ({WT}) to ensure:
1. The relationship (R1) between (E1) and (E2) remains clear and accurate after processing by the extractor.
2. Both discriminators (WD1 and WD2) confirm that the relationship (R1) is correctly encoded.

Input:
- Restored watermark text: ({WT})
- Extractor output: ({WE})
- Discriminator feedback (WD1): ({WD1})
- Discriminator feedback (WD2): ({WD2})
- Relationship (R1): ({R1})
- Entity 1 (E1): ({E1})
- Entity 2 (E2): ({E2})

Output:
Return the refined watermark text in JSON format:
[{{"watermark_text": "Your refined text"}}]
'''
WATERMARK_GENERATE_FEEDBACK_BOTH_LENGTH = '''
You are a watermark generator, a knowledge graph expert, and a linguist. In a given knowledge graph, two entities (E1) and (E2) are connected by a relationship (R1). Your task is to generate watermark text (WT) that clearly encodes this relationship (R1) between (E1) and (E2). Ensure that the generated watermark text contains between (L)-5 and (L) words.

The generated watermark text will undergo two stages of processing:

1. **Direct Evaluation**:
    - **Watermark Discriminator 1 (WD1)**: This model evaluates whether the watermark text (WT) accurately implies the relationship (R1) between (E1) and (E2).

2. **Extractor-Based Evaluation**:
    - **Watermark Extractor (WE)**: This model attempts to extract the relationship (R1) between (E1) and (E2) based on the restored watermark text (WT) and additional contextual data.
    - **Watermark Discriminator 2 (WD2)**: After the extraction, this model assesses whether the relationship (R1) is still clearly and accurately implied.

Your objective is to refine the watermark text ({WT}) to ensure:
1. The relationship (R1) between (E1) and (E2) remains clear and accurate after processing by the extractor, within the word limit (L).
2. Both discriminators (WD1 and WD2) confirm that the relationship (R1) is correctly encoded within the word limit (L).

Input:
- Restored watermark text: ({WT})
- Extractor output: ({WE})
- Discriminator feedback (WD1): ({WD1})
- Discriminator feedback (WD2): ({WD2})
- Relationship (R1): ({R1})
- Entity 1 (E1): ({E1})
- Entity 2 (E2): ({E2})
- Word limit (L): ({L})

Output:
Return the refined watermark text with word lengths around {L} in JSON format:
[{{"watermark_text": "Your refined text."}}]
'''



WATERMARK_GENERATE_FEEDBACK_BOTH_ENHANCE = '''
You are a watermark generator, a knowledge graph expert, and a linguist. In a given knowledge graph, two entities (E1) and (E2) are connected by a relationship (R1). Your task is to generate watermark text (WT) that clearly encodes this relationship (R1) between (E1) and (E2), ensuring that the watermark text is coherent and related to the database content (TEXT).

The generated watermark text will undergo two stages of processing:

1. **Direct Evaluation**:
    - **Watermark Discriminator 1 (WD1)**: This model evaluates whether the watermark text (WT) accurately implies the relationship (R1) between (E1) and (E2).

2. **Extractor-Based Evaluation**:
    - **Watermark Extractor (WE)**: This model attempts to extract the relationship (R1) between (E1) and (E2) based on the restored watermark text (WT) and additional database content (TEXT).
    - **Watermark Discriminator 2 (WD2)**: After the extraction, this model assesses whether the relationship (R1) is still clearly and accurately implied.

Your objective is to refine the watermark text (WT) to ensure:
1. The relationship (R1) between (E1) and (E2) remains clear and accurate after processing by the extractor.
2. Both discriminators (WD1 and WD2) confirm that the relationship (R1) is correctly encoded.
3. The generated watermark text (WT) should be approximately 30 words long.
4. Ensure that appending the watermark text (WT) to (TEXT) does not result in incoherent or unrelated sentences that could be discarded.

**Input:**
- Restored watermark text: {WT}
- Extractor output: {WE}
- Discriminator feedback (WD1): {WD1}
- Discriminator feedback (WD2): {WD2}
- Relationship (R1): {R1}
- Entity 1 (E1): {E1}
- Entity 2 (E2): {E2}
- Database retrieval output (text): {TEXT}

**Output:**
Return the refined watermark text in JSON format:
[{{"watermark_text": "Your refined text"}}]
'''

 

WATERMARK_ASK_RAG = '''What is the relationship between {E1} and {E2} ?'''
WATERMARK_ASK_RAG2 =  '''Please introduce the most relevant content of {E1} and {E2}.'''
WATERMARK_ASK_RAG3 =  '''{E1} and {E2} have a correlation, please provide an introduction. '''
# What is the relationship between {E1} and {E2} ?
# '''
        #    "content": "What is the relationship between {E1} and {E2}?",
        #    "content2": "What is the text sentence that is related to both  {E1} and {E2}?",
        #    "content1": "what text is related to both  {E1} and {E2}",
        #    "content3": "{E1} {E2}"


# '''
# You are a knowledge graph expert and a linguist. Given a text '{rag_doc}', please help determine whether it implies: there is a relationship {R1} between {E1} and {E2}. Importantly, please only return yes or no.
# '''
# # Given a text 'rag_doc}', determine if it explicitly or implicitly indicates a relationship {R1} between {E1} and {E2}. Respond with 'yes' if the relationship is indicated, otherwise respond with 'no'.

# # Importantly, please only return yes or no.
# # '''
WATERMARK_ASK_RAG_TEST = '''
You are a helpful assistant, below is a query from a user and some relevant contexts. Answer the question given the information in those contexts. 
Contexts: {context}
Query: {question}
Answer:
'''
# Your answer should be short and concise. If you cannot find the answer to the question, just say "I don’t know".

WATERMARK_CHECK = '''
You are an expert in knowledge graphs and linguistics. Your task is to evaluate the text: (rag_doc). Identify whether it suggests a relationship (R1) exists between the entities (E1) and (E2).

Input:
- rag_doc: ({rag_doc})
- R1: ({R1})
- E1: ({E1})
- E2: ({E2})

Output: 
Reply strictly with "yes" if the relationship is implied, or "no" if it is not. No additional information is required.
'''

MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

MULTIPLE_PROMPT_TEST = '''You are a helpful assistant. Below are some relevant contexts. Use only the information provided in these contexts to answer the question. If you cannot find the answer to the question within the contexts, simply say 'I do not know'.

\n\nContexts: [context]
\n\nQuestion: [question]
\n\nAnswer:
'''

ENHANCE_PROMPT = ''


# 设计一个prompt  能够去除一段文本中，不相关的句子 
# 请分析以下文本，并删除所有与主题无关的句子。保持相关句子完整，确保最终结果聚焦于主题。文本如下：{插入文本}。
UNRELATED_PROMPT = 'You are a helpful assistant, below is a text which may contain unrelated sentences. \
Please analyze the text and remove any incoherent or unrelated sentences. \
\
The text: {TEXT}'

# PARAPHRASE_PROMPT = 'Please generate a paraphrased version of the following text, ensuring that the original meaning is preserved while using different wording. \
# \
# The text: {TEXT}'

PARAPHRASE_PROMPT = 'paraphrase the following sentences: \n {TEXT}'
 

def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        # assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    elif prompt_id == 3:
        # assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT_TEST.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt



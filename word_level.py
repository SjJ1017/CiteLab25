import json
import re
import numpy as np

def all_normalize(obj):
    all_values = []
    for output_sent_result in obj:
        for each_doc in output_sent_result:
            for each_span in each_doc:
                all_values.append(each_span[1])
    max_val = max(all_values)
    min_val = min(all_values)
    for output_sent_result in obj:
        for i, each_doc in enumerate(output_sent_result):
            for j, each_span in enumerate(each_doc):
                each_span = (each_span[0], (each_span[1] - min_val) / (max_val - min_val))
                output_sent_result[i][j] = each_span
    return obj
    

def load_json(file_path):

    with open(file_path, 'r') as file:
        data = file.read()
    if file_path.endswith('.jsonl'):
        data = f'[{'},{'.join(data.split("}\n{"))}]'
    objects = json.loads(data)
    return objects

def ma(text):
    pattern = r"Document \[\d+\]\(Title:[^)]+\):"

    match = re.search(pattern, text)

    if match:
        index = match.end()
        return index
    else:
        return 0

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def split_by_docs(scores, docs_text, doc_tokens):
    assert len(scores) == len(doc_tokens)
    sep = '\n\n'
    docs = docs_text.strip().split(sep)
    doc_lens = [len(doc) for doc in docs]
    doc_end_idx = [sum(doc_lens[:i+1]) for i in range(len(doc_lens))]
    print(doc_end_idx)

    last_tokens = [0]
    for i, token in enumerate(doc_tokens):
        next_token = doc_tokens[i+1] if i+1 < len(doc_tokens) else None
        if token == "<0x0A>" and next_token == "<0x0A>": # FOR LLAMA2 ONLY
            last_tokens.append(i + 1)
    for i, idx in enumerate(last_tokens[1:]):
        pre_idx = last_tokens[i]
        curr_tokens = doc_tokens[pre_idx:idx + 1]
        curr_tokens = [token for token in curr_tokens if token != "<0x0A>"]
        curr_doc = ''.join(curr_tokens)
        while curr_doc.startswith('\u2581'):
            curr_doc = curr_doc[1:]
        #print(curr_doc)
        #print(docs[i])
        #assert len(curr_doc) == len(docs[i]), f"{len(curr_doc)} != {len(docs[i])}"
    doc_num = len(last_tokens) - 1
    scores_per_doc = [[] for _ in range(doc_num)]
    curr_doc_idx = 0
    skip = False
    curr_char_idx = -2 # magic number
    for i, (score, token) in enumerate(zip(scores, doc_tokens)):
        if skip:
            skip = False
            continue
        if i == 0:
            token = token[1:] # remove the first space
        if token == "<0x0A>":
            curr_doc_idx += 1
            curr_char_idx = -2
            skip = True # skip the next token
            continue
        scores_per_doc[curr_doc_idx].append((curr_char_idx, score))
        curr_char_idx += len(token)
    #print(scores_per_doc[0])
    for i, doc in enumerate(docs):
        start = ma(doc) - 2
        #print(start)
        scores_per_doc[i] = list(filter(lambda x: x[0] >= start, scores_per_doc[i]))
    all_values = []
    for scores in scores_per_doc:
        # normalize
        all_values.extend([score[1] for score in scores])
    max_val = max(all_values)
    min_val = min(all_values)
    for scores in scores_per_doc:
        for i, score in enumerate(scores):
            scores[i] = (score[0], (score[1] - min_val) / (max_val - min_val))
            
    return scores_per_doc

def span_to_doc(results):
    for res in results:
        span_level = res['span_level']
        doc_level = []
        for output_sent_result in span_level:
            doc_level.append([np.mean([span[1] for span in doc]) for doc in output_sent_result])
        res['doc_level'] = doc_level
    return results




def word_level_attribute(raw, _i):
    res = load_json(f'MIRAGE/internal_res/res_attr_dict-{_i}.json')

    input_text = res["input_context"]
    input = res["input_context_tokens"]
    output = res["output_current"]
    output_tokens =res["output_current_tokens"]
    token_lens = [len(x) for x in output_tokens]
    cci_scores = res["cci_scores"]
    splited_output = raw[_i]["output"]
    all_lens = [len(x) for x in splited_output]
    end_token_idx = [sum(token_lens[:i+1]) for i in range(len(token_lens))]
    end_idx = [sum(all_lens[:i+1]) for i in range(len(all_lens))]
    end_idx = [len(list(filter(lambda x: x < idx, end_token_idx))) for idx in end_idx]
    belong_sents = [[] for _ in range(len(splited_output))]
    for token_cci in cci_scores:
        token_idx = token_cci['cti_idx']
        for i, idx in enumerate(end_idx):
            if token_idx < idx:
                belong_sents[i].append(token_cci)
                break
    scores = []
    for i, sent in enumerate(belong_sents):
        weighted_scores = [token_cci["cti_score"]*np.array(token_cci["input_context_scores"]) for token_cci in sent]
        #weighted_scores = [np.array(token_cci["input_context_scores"]) for token_cci in sent]
        sum_scores = np.sum(weighted_scores, axis=0)
        #max_scores = np.max(weighted_scores, axis=0)
        scores.append(sum_scores)
        #scores.append(max_scores)
    finals = []

    for score in scores:
        doc_scores = split_by_docs(score, input_text, input)
        finals.append(doc_scores)


    doc_finals = [[] for _ in range(len(finals))]
    for i, output_sent_result in enumerate(finals):
        docs = []
        for doc in output_sent_result:
            doc_score = sum([score[1] for score in doc])
            docs.append(doc_score)
        doc_finals[i] = docs
    print(doc_finals)



    raw[_i]["word_level"] = finals
    raw[_i]["doc_level"] = doc_finals

raw = load_json('results.json')
for i in range(len(raw)):
    word_level_attribute(raw, i)
write_json('result_.json', raw)
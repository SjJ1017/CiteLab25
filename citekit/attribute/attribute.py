import json
from context_cite import ContextCiter
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


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

def all_normalize_in(obj):
    for output_sent_result in obj:
        all_values = []
        for each_doc in output_sent_result:
            for each_span in each_doc:
                all_values.append(each_span[1])
        max_val = max(all_values)
        min_val = min(all_values)
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
    pattern = r"Document \[\d+\]\(Title:[^)]+\)"

    match = re.search(pattern, text)

    if match:
        index = match.end()
        return index
    else:
        return 0

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)



def load_model(model_name_or_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        token = 'your token'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    return model, tokenizer


def compute_log_prob(model, tokenizer, input_text, output_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output_tokens = tokenizer(output_text, return_tensors="pt")["input_ids"]
    
    with torch.no_grad():
        logits = model(**inputs).logits[:, -output_tokens.shape[1]-1:-1, :]
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    output_log_probs = log_probs.gather(2, output_tokens.unsqueeze(-1)).squeeze(-1)
    return output_log_probs.sum().item()

def compute_contributions(model, tokenizer, question, docs, output):
    full_input = question + '\n\n' + '\n'.join(docs)
    base_prob = compute_log_prob(model, tokenizer, full_input, output)
    
    contributions = []
    for i in range(len(docs)):
        reduced_docs = docs[:i] + docs[i+1:]
        reduced_input = question + '\n\n' + '\n'.join(reduced_docs)
        reduced_prob = compute_log_prob(model, tokenizer, reduced_input, output)
        contributions.append(base_prob - reduced_prob)
    
    return contributions
    
class InterpretableAttributer:

    def __init__(self, levels=['doc', 'span', 'word'], model = 'gpt-2'):
        for level in levels:
            assert level in ['doc', 'span', 'word'], f'Invalid level: {level}'
        # span before doc
        self.levels = sorted(levels, key=lambda x: ['span', 'doc', 'word'].index(x))
        #self.model, self.tokenizer = load_model(model)


    def attribute(self, question, docs, output):
        attribute_results = {}
        for level in self.levels:
            attribute_result = []
            for sentence in output:
                attribute_result.append(self._attribute(question, docs, sentence, level))
            attribute_results[level] = attribute_result
        return attribute_results
    

    def _attribute(self, question, docs, output, level):
        if level == 'doc':
            return self.doc_level_attribution(question, docs, output)
        elif level == 'span':
            return self.span_level_attribution(question, docs, output)
        elif level == 'word':
            return self.word_level_attribution(question, docs, output)
        else:
            raise ValueError(f'Invalid level: {level}')
    
    def span_level_attribution(self, question, docs, output):
        # USE CONTEXT CITE
        context = '\n\n'.join(docs)
        response = output

        cc = ContextCiter(self.model, self.tokenizer, context, question)
        _, prompt = cc._get_prompt_ids(return_prompt=True)
        cc._cache["output"] = prompt + response  
        result = cc.get_attributions(as_dataframe=True, top_k=1000).data.to_dict(orient='records')
        return result
    
    
    def parse_attribution_results(self, docs, results):
        context = '\n\n'.join(docs)
        lens = [len(doc) for doc in docs]
        len_sep = len('\n\n')
        final_results = {}
        for level, result in results.items(): 
            if level == 'span':
                ordered_all_sents = []
                for output_sent_result in result:
                    final_end_for_span = {}
                    all_span_results = []
                    for each_span in output_sent_result:
                        span_text = each_span["Source"]
                        span_score = each_span["Score"]
                        start = 0
                        if span_text in final_end_for_span:
                            start = final_end_for_span[span_text]
                        span_start = context.find(span_text, start)
                        span_end = span_start + len(span_text)
                        final_end_for_span[span_text] = span_end
                        # locate the document
                        doc_idx = 0
                        while span_start > lens[doc_idx]:
                            span_start -= lens[doc_idx] + len_sep
                            span_end -= lens[doc_idx] + len_sep
                            doc_idx += 1
                        all_span_results.append((span_start, span_score, doc_idx))
                    ordered = [[] for _ in range(len(docs))]
                    for span_start, span_score, doc_idx in all_span_results:
                        ordered[doc_idx].append((span_start, span_score))
                    for i in range(len(docs)):
                        doc = docs[i]
                        real_start = ma(doc)
                        ordered[i] = sorted(ordered[i], key=lambda x: x[0])
                        ordered[i][0] = (real_start, ordered[i][0][1])

                    ordered_all_sents.append(ordered)
                final_results[level+'_level'] = all_normalize_in(ordered_all_sents)
            elif level == 'doc':
                self.span_to_doc(result)
            else:
                raise NotImplementedError(f'Parsing for {level} not implemented yet')
        return final_results
    
    def span_to_doc(self, results):
        import numpy as np
        span_level = results['span_level']
        doc_level = []
        for output_sent_result in span_level:
            doc_level.append([np.mean([span[1] for span in doc]) for doc in output_sent_result])
        results['doc_level'] = doc_level
        

    def attribute_for_result(self, result):
        docs = result['doc_cache']
        question = result['data']['question']
        output = result['output']
        attribution_results = self.attribute(question, docs, output)
        parsed_results = self.parse_attribution_results(docs, attribution_results)
        result.update(parsed_results)

        if 'doc' not in self.levels:
            # if doc is not in the levels, we need to convert the span level to doc level
            print('Converting span level to doc level...')
            try:
                self.span_to_doc(result)
                print('Conversion successful')
            except Exception as e:
                print(f'Error converting span level to doc level: {e}')

    def attribute_for_results(self, results):
        for result in results:
            self.attribute_for_result(result)
        return results
    

if __name__ == '__main__':
    attributer = InterpretableAttributer(levels=['span'])
    results = load_json('res_attr.json')
    attributer.attribute_for_results(results)
    write_json('res_attr_span.json', results)
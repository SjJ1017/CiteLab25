from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import CitationSimplyfier
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT,PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEDocPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator
from parser import *
import argparse
import json
from citekit.utils.utils import cut_and_make_as,one_paragraph,make_as
from nltk import sent_tokenize


def sentences_as(datakey):
    def f(passage):
        print( sent_tokenize(passage))
        return [{datakey:one_paragraph(s)} for s in sent_tokenize(passage)]
    return f

PARA_SEP = '\n\n'
if __name__ == '__main__':

    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=1, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=3, help="number of docs")
    parser.add_argument("--pr", action='store_true', help="use cite PR")
    parser.add_argument("--rouge", action='store_true', help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", action='store_true', help="eval qa")
    parser.add_argument("--str_em", action='store_true', help="eval str_em")
    parser.add_argument("--mauve",  action='store_true', help="eval mauve")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", action='store_true', help="eval claims")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--turns", type=int, default=1, help="k")
    parser.add_argument("--use_fast_pr", type=str, default=False, help="test")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/AnG.json', help="demo")
    parser.add_argument("--mode", type=str, default='plan', help="mode: AnG or plan")
    parser.add_argument("--data_num", type=int, default=200, help="k")
    args = parser.parse_args()

    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo
    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)[args.mode]

    answer_inst = 'Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly, by answering the subquestion. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
    revise_instruction = '''You will be provided with a question, some documents and an answer. Your task is to correct possible errors in the answer and made it more coherent and fluent. Only make changes that are necessary. Keep the citations.'''
    dataset = PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data_wo_title(data['docs'][:args.ndoc]))[:args.data_num]
    if args.mode == 'AnG':
        gen_shot = demo['gen_instruction'] + PARA_SEP + demo['gen_shot'] + PARA_SEP
        answer_ppt = {'INST':demo['gen_instruction'],'shot':gen_shot, 'add':'The next sentence is:'}
    elif args.mode == 'plan':
        shot = demo['shot1'] + demo['shot2']
        self_ppt = {'INST':demo['INST'],'shot':shot, 'add':'subquestions: \n'}
        answer_shot = demo['answer_shot_1'] + demo['answer_shot_2']
        answer_ppt = {'INST':answer_inst,'shot':answer_shot,'add':''}

    prompt = Prompt(template='<shot><INST><question><docs><prefix><sub><span><add><new_ans>',
                    components={'INST':'{INST}\n\n', 
                                'shot':'{shot}',
                                'question':'Question:{question}\n\n',
                                'docs':'{docs}\n',
                                'span':'The highlighted spans are: \n{span}\n\n',
                                'prefix':'Prefix: {prefix}\n\n',
                                'sub':'subquestions: \n{sub}\n\n',
                                'add':'Answer: \n{add}',
                                'new_ans': "\nNew Answer: {new_ans}"
                                })
    
    plan_prompt = Prompt(template='<shot><INST><question><docs><sub><add>',
                    components={'INST':'{INST}\n\n', 
                                'shot':'{shot}',
                                'question':'Question:{question}\n\n',
                                'docs':'{docs}\n',
                                'sub':'subquestions: \n{sub}\n\n',
                                'add':'{add}'})

    # PIPELINE
    evaluator = DefaultEvaluator(args)

    attribute = LLM(model = args.model, prompt_maker = plan_prompt, self_prompt=self_ppt, post_processing=sentences_as('sub'))
    reviser = LLM(model=args.model, prompt_maker=prompt, self_prompt={'INST':revise_instruction, 'new_ans': ''}, max_turn=args.turns)
    answer = LLM(model = args.model, prompt_maker = prompt, self_prompt=answer_ppt, share_model_with=attribute.get_first_module(), merge=True, parallel=True)

    # No citation -> simply combine
    # simplifier
    answer.set_target(reviser, post_processing=lambda output: {'add': output})
    attribute.set_target(answer,post_processing=sentences_as('sub'))
    pipeline = Pipeline(save_path=args.save_path, llm = answer, module = [attribute, reviser], evaluator = evaluator, dataset = dataset)
    pipeline.set_initial_module(module=attribute)
    answer.set_output(post_processing=lambda ls: ''.join(map(one_paragraph,ls)))




    #pipeline.run_on_dataset(datakeys=['question','docs'], init_docs='docs', initial_module = attribute)

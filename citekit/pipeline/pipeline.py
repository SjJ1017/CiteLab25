from citekit.cite_modules.LLM import LLM,Module
from citekit.cite_modules.augment_model import AugmentCluster, AttributingModule, MODEL_TYPE_MAPPING
from citekit.prompt.prompt import ALCEVanillaPrompt, DocPrompt
import logging
import json
from tqdm import tqdm
import traceback
import copy
from citekit.utils.utils import flatten_dict
import csv



def merge_str_dicts(dicts):
    result = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in result:
                result[key] += ' ' + value
            else:
                result[key] = value
    return result

PIPELINE_OUTPUT = 'output'
PIPELINE_DOC_CACHE = 'doc_cache'

class DocCache():
    def __init__(self) -> None:
        self.__docs = list()
    
    def __len__(self):
        return len(self.__docs)

    def __getitem__(self,index):
        if index>=0 and index <len(self):
            return self.__docs[index]
        else:
            return None
    
    def get_last(self):
        if self.__docs:
            return self.__docs[-1]
        
    def add_doc(self, doc, add_id = True) -> int:
        if not isinstance(doc, str):
            assert isinstance(doc, dict) and 'text' in doc and 'title' in doc
            doc = f'(Title: {doc["title"]}){doc["text"]}'
        if add_id:
            doc_head = f'Document [{len(self)+1}]'
        else:
            doc_head = ''
        self.__docs.append(doc_head + doc)
        return len(self)

    def load_docs(self, docs, add_id = False):
        for doc in docs:
            self.add_doc(doc, add_id)
        return len(self)
    
    def clear(self):
        self.__docs = list()

    def show_docs(self):
        return self.__docs


class Pipeline():
    def __init__(self,save_path = None, sequence = None, head_prompt_maker = None, llm = None, module= None, retriever = None, evaluator = None, dataset = None, rich_eval = False, train_data = False, attributer = None) -> None:
        self.save_path = save_path
        self.train_data = train_data
        self.head_prompt_maker = head_prompt_maker
        self.table_head = True
        self.attributer = attributer
        self.llm = llm
        self.initial_docs = None
        self.data_keys = None
        self.stored_clusters = []
        self.module = []
        if llm:
            llm.connect_to(self)
        if not isinstance(module,list) and module is not None:
            if module:
                module.connect_to(self)
        else:
            if isinstance(module, list):
                for i in module:
                    if isinstance(i, AugmentCluster) or isinstance(i, Module):
                        i.connect_to(self)
        self.dataset = dataset

        self.data_index = 0
        self.retriever = retriever
        if retriever:
            retriever.pipeline = self

        self.eval = evaluator
        if evaluator:
            evaluator.pipeline = self
        self.output = []
        self.log = []
        self.doc_cache = DocCache()
        self.head = {}
        self.result = {}
        self.rich_eval = rich_eval
        self.initial_module = None
        
    def load_data(self, dataset):
        self.data = dataset

    def set_initial_module(self, module):
        self.initial_module = module
    def get_initial_module(self):
        return self.initial_module
    
    def set_data_keys(self, keys):
        self.data_keys = keys
    def get_data_keys(self):
        return self.data_keys
    

    def update(self, update_object, config, update_info):
        print(f'Updating {update_object} with {config} and {update_info}')
        module = self.get_module_by_name(update_object)
        if config in ['prompt', 'header']:
            module.update(config, update_info)
        elif config in ['destination']:
            module.update(config, [self.get_module_by_name(update_info[0]), update_info[1]])
        elif config in ['delete_destination']:
            module.update(config, self.get_module_by_name(update_info))
        elif config in ['new_model']:
            model_type, model, key = update_info
            print('Creating new model:', model_type, model)
            new_model_class = MODEL_TYPE_MAPPING[model_type]
            print('New model class:', new_model_class)
            new_model = new_model_class(model)
            new_model.connect_to(self)
            print('Created new model:', new_model)
            module.update('destination', [new_model, key])

        else:
            raise NotImplementedError



    def set_initial_docs(self, d):
        self.initial_docs = d
    def get_initial_docs(self):
        return self.initial_docs
    
    def run_on_dataset(self,datakeys,init_docs=None,initial_module= None,start=0):
        if self.initial_module and not initial_module:
            initial_module = self.initial_module
        if self.save_path:
            for i in range(start,len((self.dataset))):
                self.data_index = i
                try:
                    self.run(datakeys,init_docs,initial_module,train=self.train_data)
                except Exception as e:
                    print(f'Error: {e}, skipping data {i}')
                    traceback.print_exc()
        else:
            for i in range(start,len((self.dataset))):
                self.data_index = i
                try:
                    self.run(datakeys,init_docs,initial_module,write=False,train=self.train_data)
                except Exception as e:
                    print(f'Error: {e}, skipping data {i}')
                    traceback.print_exc()
        
            

    def form_eval_data(self) -> dict:
        """To write rich eval, you can use data from:
        pipeline.dataset, doc_cache and output 
        to post_process data as a argument dict for evaluation
        """
        raise NotImplementedError('You have to write <form_eval_data function> to apply rich eval with designed arguments.')

    def direct_run(self, dynamic_prompt= {}, module = None):
        if not module:
            module = self.llm
        if isinstance(module, AugmentCluster):
            module = module.get_first_module()
        while isinstance(module, Module):
            if isinstance(dynamic_prompt,dict):
                module.change_to_multi_process(False)
                dynamic_prompt = module.generate(self.head,dynamic_prompt=dynamic_prompt)
            elif isinstance(dynamic_prompt,list) and all([isinstance(d,dict) for d in dynamic_prompt]):
                module.change_to_multi_process(True)
                if module.parallel:
                    dynamic_prompt = [module.generate(self.head,d) for d in dynamic_prompt]
                    if module.merge:
                        dynamic_prompt = merge_str_dicts(dynamic_prompt)
                    module.add_output_to_head(module.last_message)
                elif not module.iterative and not module.merge:
                    for d in dynamic_prompt:
                        self.direct_run(dynamic_prompt = d, module = copy.copy(module))
                    #dynamic_prompt = [module.generate(self.head,d) for d in dynamic_prompt]
                    break
                elif module.iterative:
                    iter_dynamic = {}
                    for d in dynamic_prompt:
                        iter_dynamic = module.generate(self.head,{**d,**iter_dynamic})
                    dynamic_prompt = iter_dynamic
                module.end_multi()
            else:
                print(type(dynamic_prompt))
                raise TypeError(str(dynamic_prompt))
            self.log.append(f'{module} -> {module.send()}\n: {module.last_message}')
            if isinstance(module, Module):
                module.output()
                print('DEBUG:', str(module), module.end, module.turns, module.max_turn)
                if module.end or module.turns > module.max_turn:
                    break
            module = module.send()
            if isinstance(module, AugmentCluster):
                module = module.get_first_module()


    def __call__(self, data):
        # run only one data
        # backup
        dataset_backup = self.dataset
        current_data_index_backup = self.data_index
        if hasattr(self,'current_data'):
            current_data_backup = self.current_data
        else:
            current_data_backup = None

        # set data and run
        dataset = [data]
        self.dataset = dataset
        self.data_index = 0
        result = self.run(datakeys = self.data_keys, init_docs = self.initial_docs, initial_module = self.initial_module, write = False, train = False)

        # restore
        self.data_index = current_data_index_backup
        self.current_data = current_data_backup
        self.dataset = dataset_backup


        return result

    def run(self, datakeys, init_docs = None, initial_module = None, write = True, train = False):
        
        # get data
        self.current_data = self.dataset[self.data_index]
        data = self.current_data

        # from head prompt from specific data
        head = dict()
        for key in datakeys:
            if isinstance(data[key],str):
                head[key] = data[key]
            else:
                assert isinstance(data[key],list)
                assert all([isinstance(item, str) for item in data[key]])
                head[key] = ''.join(data[key])

        #init
        self.head = head
        self.output = []
        self.doc_cache.clear()
        if init_docs:
            self.doc_cache.load_docs(data[init_docs])
        self.llm.reset()
        if self.module:
            for i in self.module:
                i.reset()
        self.log = []
        # run only one data, and add data_index by 1
        dynamic_prompt = {}
        if not initial_module:
            module = self.llm
        else:
            module = initial_module
        if isinstance(module, AugmentCluster):
            module = module.get_first_module()
        while isinstance(module, Module):
            if isinstance(dynamic_prompt,dict):
                module.change_to_multi_process(False)
                dynamic_prompt = module.generate(self.head,dynamic_prompt=dynamic_prompt)
            elif isinstance(dynamic_prompt,list) and all([isinstance(d,dict) for d in dynamic_prompt]):
                module.change_to_multi_process(True)
                if module.parallel:
                    dynamic_prompt = [module.generate(self.head,d) for d in dynamic_prompt]
                    if module.merge:
                        dynamic_prompt = merge_str_dicts(dynamic_prompt)
                    module.add_output_to_head(module.last_message)
                elif not module.iterative and not module.merge:
                    for d in dynamic_prompt:
                        self.direct_run(dynamic_prompt = d, module = copy.copy(module))
                    #dynamic_prompt = [module.generate(self.head,d) for d in dynamic_prompt]
                    break

                elif module.iterative:
                    iter_dynamic = {}
                    for d in dynamic_prompt:
                        iter_dynamic = module.generate(self.head,{**d,**iter_dynamic})
                    dynamic_prompt = iter_dynamic
                module.end_multi()
            else:
                print(type(dynamic_prompt))
                raise TypeError(str(dynamic_prompt))
            self.log.append(f'{module} -> {module.send()}\n: {module.last_message}')
            if isinstance(module, Module):
                module.output()
                if module.end or module.turns > module.max_turn:
                    break
            module = module.send()
            if isinstance(module, AugmentCluster):
                module = module.get_first_module()

        # if eval, send to evaluation
        if self.eval: 
            if not self.rich_eval:
                self.result = self.eval()
            else:
                self.result = self.eval(self.form_eval_data())
        else:
            self.result = {}
        if write:
            self.write()
        if train:
            self.export_training_data()
        
        #self.logs = self.delete_inner_cluster_logs(self.log)
        res = {'data':self.get_data(), 'doc_cache':self.doc_cache.show_docs(), 'log': self.log.copy(),'output':self.output,'result': self.result}
        if self.attributer:
            self.attributer.attribute_for_result(res)
        self.data_index += 1
        return res
    
    def delete_inner_cluster_logs(self, logs):
        print(logs)
        for cluster in self.stored_clusters:
            cluster_name = str(cluster)
            print('Combining logs for cluster:', cluster_name)
            in_cluster = False
            for i, log in enumerate(logs):
                in_out_names = log.split('\n')[0]
                if in_out_names in cluster_name:
                    # This is the inner log
                    if not in_cluster:
                        in_cluster = True
                        log_start = i
                    else:
                        continue
                elif in_cluster:
                    # This is the outer log
                    in_cluster = False
                    log_end = i
                    cluster_output = logs[log_end]
                    next_module = in_out_names.split('->')[1].strip()
                    cluster_log = f"{cluster_name} -> {next_module}\n: {cluster_output}"
                    logs = logs[:log_start] + [cluster_log] + logs[log_end+1:]
        print('Final logs:', logs)
        return logs
        
                    


    def get_data(self):
         return self.dataset[self.data_index]

    def write(self):
        '''Default writing'''
        llm_token_used = self.llm.token_used
        write_down = {'data':self.get_data(), 'doc_cache':self.doc_cache.show_docs(), 'log': self.log.copy(),'output':self.output,'result': self.result,'token_used':llm_token_used}

        if self.attributer:
            self.attributer.attribute_for_result(write_down)

        with open(self.save_path, 'a', encoding='utf-8') as file:
            json_line = json.dumps(write_down, indent=4)
            file.write(json_line + '\n')

    def get_module_by_name(self, name):
        print('Getting module by name:', name)
        for module in self.module:
            if str(module) == name:
                return module
        if str(self.llm) == name:
            return self.llm
        
        for cluster in self.stored_clusters:
            print('trying cluster:', cluster)
            if str(cluster) == name:
                print('found cluster:', cluster)
                return cluster
    
        return None
    
    def export_training_data(self):
        flattened_data = [flatten_dict(self.result)]
        header = set()
        for item in flattened_data:
            header.update(item.keys())
        header = sorted(header)
        with open('output.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = header)
            if self.table_head:
                writer.writeheader()
                self.table_head = False
            
            for row in flattened_data:
                writer.writerow(row)


    def __str__(self) -> str:
        return 'pipeline output'

class Sequence(Pipeline):
    def __init__(self, save_path=None, sequence=None, head_prompt_maker=None, retriever=None, evaluator=None, dataset=None, rich_eval=False) -> None:
        first_module = sequence[0]
        other = sequence[1:]
        super().__init__(save_path, sequence, head_prompt_maker, first_module, other, retriever, evaluator, dataset, rich_eval)
        for i in range(len(sequence)-1):
            module = sequence[i]
            assert isinstance(module, Module) or isinstance(module,AugmentCluster)
            module.set_target(sequence[i+1],post_processing=lambda x: {module.output_as: x})
        sequence[-1].set_output()
        

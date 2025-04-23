import hashlib
import copy
import json
from citekit.prompt.prompt import Prompt, ALCEDocPrompt,DocPrompt,NewALCEVanillaPrompt

try:
    example_results = json.load(open('res.json'))
except:
    example_results = []


def weight_check(module, target):
    if module.model_type == 'verifier':
        test_module = copy.deepcopy(module)
        test_module.last_message = True
        true_target = test_module.send()
        if str(target) == str(true_target) or ('output' in str(target).lower() and 'output' in str(true_target).lower()):
            return 1
        test_module.last_message = False
        true_target = test_module.send()
        if str(target) == str(true_target) or ('output' in str(target).lower() and 'output' in str(true_target).lower()):
            return -1
        
    return 0

def get_params(module):
    import copy
    componets = copy.deepcopy(module.prompt_maker.components if hasattr(module, 'prompt_maker') and hasattr(module.prompt_maker, 'components') else None)
    if componets:
        if isinstance(componets, tuple):
            componets = componets[0]
        componets.update(module.self_prompt)
    destination = module.get_destinations()
    if len(destination) == 1:
        destination = str(destination[0])
    elif not destination:
        destination = 'N/A'
    else:
        destination = 'Not Available for multiple destinations'

    if hasattr(module, 'iterative'):
        mode = 'iterative' if module.iterative else ('parallel' if module.parallel else None) 
    else:
        mode = None
    
    if hasattr(module, 'if_add_output_to_head') and hasattr(module, 'head_key'):
        global_prompt = 'N/A' if module.if_add_output_to_head == False else module.head_key
    else:
        global_prompt = 'N/A'


    params = {
        'type': module.model_type,
        'Model': getattr(module, 'model_name', None),
        'Mode': mode,
        'Max Turn': getattr(module, 'max_turn', None),
        'Destination': destination,
        'Prompt': module.prompt_maker.make_prompt(componets) if hasattr(module,'prompt_maker') and module.prompt_maker else None,
        'Global Prompt': global_prompt,
    }
    print(params)
    non_empty_params = {}
    for k, v in params.items():
        if v:
            non_empty_params[k] = v
    return non_empty_params

class PipelineGraph:

    def __init__(self, pipeline = None):
        self.pipeline = pipeline
        self.nodes = {'input': {}, 'output': {}}
        self.edges = {}  # 邻接表：{from_node: {to_node: weight}}
        self.node_count = 0
        if pipeline:
            self.load_pipeline(pipeline)
            print('FINDING ATTR:', [str(module) for module in pipeline.module])

    def get_auto_node_name(self):
        """生成自动节点名。"""
        self.node_count += 1
        return f"Node{self.node_count}"
    
    def update(self):
        self.__init__(pipeline=self.pipeline)

    def load_pipeline(self, pipeline):
        """从现有Pipeline对象加载图。"""
        initial_module = pipeline.get_initial_module()
        processed_nodes = []
        if not initial_module:
            initial_module = pipeline.llm
        self.add_node(name = str(initial_module), 
                      params = get_params(module=initial_module))
        self.add_edge(from_node='input', to_node=str(initial_module), weight=0)
        processed_nodes.append(initial_module)
        subnodes = initial_module.get_destinations()
        if not subnodes:
            self.add_edge(from_node=str(initial_module), to_node='output', weight=0)
            return
        # print([str(subnode) for subnode in subnodes])
        for subnode in subnodes:
            weight = weight_check(initial_module, subnode)
            self.load_node(subnode, processed_nodes=processed_nodes)
            self.add_edge(from_node=str(initial_module), to_node=str(subnode), weight=0)
        if hasattr(initial_module,'output_cond') and initial_module.output_cond:
            weight = weight_check(initial_module, 'output')
            self.add_edge(from_node=str(initial_module), to_node='output', weight=weight)
            

    def load_node(self, module, processed_nodes = None):
        if module in processed_nodes:
            return
        self.add_node(name = str(module),
                      params = get_params(module=module))
        processed_nodes.append(module)
        subnodes = module.get_destinations()
        if not subnodes:
            self.add_edge(from_node=str(module), to_node='output', weight=0)
            return
        for subnode in subnodes:
            self.load_node(subnode, processed_nodes)
            weight = weight_check(module, subnode)
            self.add_edge(from_node=str(module), to_node=str(subnode), weight=weight)
        if hasattr(module,'output_cond') and module.output_cond:
            weight = weight_check(module, 'output')
            self.add_edge(from_node=str(module), to_node='output', weight=0)
            

    def export(self):
        """将图结构导出为字典格式，包含节点和边的信息。"""
        nodes_export = {name: params.copy() for name, params in self.nodes.items()}
        edges_export = []
        for from_node, connections in self.edges.items():
            for to_node, weight in connections.items():
                edges_export.append({
                    'from': from_node,
                    'to': to_node,
                    'weight': weight
                })
        return {
            'nodes': nodes_export,
            'edges': edges_export
        }

    @classmethod
    def import_from_dict(cls, data):
        """从字典数据创建新的PipelineGraph实例。"""
        graph = cls()  # 创建新实例，此时包含默认的input/output节点
        # 清空默认节点（假设导入数据包含完整节点信息）
        graph.nodes = {}
        # 导入节点
        for name, params in data['nodes'].items():
            graph.add_node(name, params)
        # 导入边
        for edge in data['edges']:
            graph.add_edge(edge['from'], edge['to'], edge['weight'])
        return graph

    def add_node(self, name, params=None):
        """添加或更新节点，参数可选。"""
        if params is None:
            params = {}
        self.nodes[name] = params.copy()

    def remove_node(self, name):
        """删除节点，自动移除相关边。不能删除input/output节点。"""
        if name in ['input', 'output']:
            raise ValueError("Cannot remove 'input' or 'output' nodes.")
        if name not in self.nodes:
            return
        del self.nodes[name]
        # 移除该节点作为起点的边
        if name in self.edges:
            del self.edges[name]
        # 移除其他节点到该节点的边
        for from_node in list(self.edges.keys()):
            if name in self.edges[from_node]:
                del self.edges[from_node][name]
                # 若该节点无其他出边，删除空字典以保持整洁
                if not self.edges[from_node]:
                    del self.edges[from_node]

    def add_edge(self, from_node, to_node, weight):
        """添加或更新边，权重必须是0、-1或+1。"""
        if weight not in {0, -1, 1}:
            raise ValueError("Weight must be 0, -1, or 1.")
        if from_node not in self.nodes:
            raise KeyError(f"Node '{from_node}' does not exist.")
        if to_node not in self.nodes:
            raise KeyError(f"Node '{to_node}' does not exist.")
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node][to_node] = weight

    def remove_edge(self, from_node, to_node):
        """删除指定边。"""
        if from_node in self.edges and to_node in self.edges[from_node]:
            del self.edges[from_node][to_node]
            # 清理空字典
            if not self.edges[from_node]:
                del self.edges[from_node]

    def get_nodes(self):
        """返回所有节点名。"""
        return list(self.nodes.keys())

    def get_edges(self):
        """返回所有边的列表，形式为(from, to, weight)。"""
        edges = []
        for from_node in self.edges:
            for to_node, weight in self.edges[from_node].items():
                edges.append((from_node, to_node, weight))
        return edges

    def update_node_params(self, name, params):
        """更新节点的参数表。"""
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist.")
        self.nodes[name].update(params.copy())

    def visualize(self, use_graphviz=True):
        """
        可视化图结构：
        - 默认尝试用graphviz生成矢量图（需安装graphviz）
        - 若未安装graphviz，则用文本模式输出
        """
        # 尝试用graphviz生成专业图表
        if use_graphviz:
            try:
                import graphviz
            except ImportError:
                print("Graphviz未安装，切换到文本模式。安装命令：pip install graphviz")
                return self._visualize_text()
            
            # 创建有向图，设置排版方向为从左到右
            dot = graphviz.Digraph(comment='Pipeline Graph', graph_attr={'rankdir': 'LR'})
            
            # 添加所有节点（包含参数信息）
            for node_name in self.nodes:
                params = self.nodes[node_name]
                # 将参数字典转换为易读字符串（例如 "size=2, color=red"）
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                
                # 特殊样式标记input/output节点
                if node_name in ('input', 'output'):
                    dot.node(
                        node_name, 
                        label=f"{node_name}\n({params_str})" if params_str else node_name,
                        shape='doublecircle',  # 双圆表示端点
                        style='filled',        # 填充颜色
                        fillcolor='#e6f3ff'   # 浅蓝色背景
                    )
                else:
                    dot.node(
                        node_name, 
                        label=f"{node_name}\n({params_str})" if params_str else node_name,
                        shape='box',           # 方框表示普通节点
                        style='rounded,filled',# 圆角填充
                        fillcolor='#f0f0f0'   # 浅灰色背景
                    )
            
            # 添加所有带权重的边
            for from_node, connections in self.edges.items():
                for to_node, weight in connections.items():
                    dot.edge(
                        from_node, 
                        to_node, 
                        label=str(weight),    # 显示权重
                        color='#666666',      # 边颜色
                        fontcolor='#ff5555'   # 权重文字颜色
                    )
            
            # 生成并自动打开可视化文件（Mac默认用Preview打开PDF）
            dot.render('pipeline_graph', view=True, cleanup=True)
            return
        
        # 文本模式回退方案
        self._visualize_text()
    
    def _visualize_text(self):
        """纯文本模式的可视化（备用方案）"""
        print("="*40 + "\nPipeline Graph 文本视图\n" + "="*40)
        
        # 列出所有节点及其参数
        print("\n[节点列表]")
        for node, params in self.nodes.items():
            param_desc = " ".join([f"{k}={v}" for k, v in params.items()])
            print(f"· {node:10} {param_desc}")
        
        # 列出所有边及其权重
        print("\n[边列表]")
        if not self.edges:
            print("（暂无边连接）")
        else:
            for from_node, connections in self.edges.items():
                for to_node, weight in connections.items():
                    arrow = {
                        1: "──(+)──>",   # +1权重用(+)表示
                        -1: "──(-)──>",  # -1权重用(-)表示
                        0: "───────>"    # 0权重无特殊标记
                    }[weight]
                    print(f"{from_node:10} {arrow} {to_node}")
        print("="*40 + "\n")



    def generate_html(self, results = example_results):
        assert isinstance(results, str)

        import json
        result_path = results
        results = json.load(open(results))
        """生成交互式可视化HTML，使用D3.js力导向图布局"""
        nodes_data = []
        for name in self.nodes:
            params = self.nodes[name]
            nodes_data.append({
                "id": name,
                "type": params.get("type", name),
                "params": params
            })

        edges_data = []
        for from_node, to_nodes in self.edges.items():
            for to_node, weight in to_nodes.items():
                edges_data.append({
                    "source": from_node,
                    "target": to_node,
                    "weight": weight
                })

        nodes_js = json.dumps(nodes_data)
        edges_js = json.dumps(edges_data)

        with open('htmls/html_templates/pipeline_template.txt', 'r', encoding='utf-8') as f:
            template = f.read()
        
        template = template.replace('<NODE_JS>', nodes_js).replace('<EDGE_JS>', edges_js).replace('<FILE_PATH>', result_path).replace('<OPTIONS>', "".join(f'<option value="{i}">Result {i+1}</option>' for i in range(len(results))))
        return template
    

    def get_nodes(self):
        nodes_data = []
        for name in self.nodes:
            params = self.nodes[name]
            nodes_data.append({
                "id": name,
                "type": params.get("type", name),
                "params": params
            })
        return json.dumps(nodes_data)

    def get_edges(self):
        edges_data = []
        for from_node, to_nodes in self.edges.items():
            for to_node, weight in to_nodes.items():
                edges_data.append({
                    "source": from_node,
                    "target": to_node,
                    "weight": weight
                })
        return json.dumps(edges_data)
    
    def get_json(self):
        return {
            'nodes': self.get_nodes(),
            'edges': self.get_edges()
        }
    
    def generate_html_embed(self, results = example_results):
        assert isinstance(results, str) or isinstance(results, list)

        import json
        result_path = results
        if isinstance(results, str):
            results = json.load(open(results))
        """生成交互式可视化HTML，使用D3.js力导向图布局"""
        nodes_data = []
        for name in self.nodes:
            params = self.nodes[name]
            nodes_data.append({
                "id": name,
                "type": params.get("type", name),
                "params": params
            })

        edges_data = []
        for from_node, to_nodes in self.edges.items():
            for to_node, weight in to_nodes.items():
                edges_data.append({
                    "source": from_node,
                    "target": to_node,
                    "weight": weight
                })

        nodes_js = json.dumps(nodes_data)
        edges_js = json.dumps(edges_data)

        with open('htmls/html_templates/pipeline_template_emb.txt', 'r', encoding='utf-8') as f:
            template = f.read()
        
        template = template.replace('<NODE_JS>', nodes_js).replace('<EDGE_JS>', edges_js).replace('<RESULTS>', json.dumps(results)).replace('<OPTIONS>', "".join(f'<option value="{i}">Result {i+1}</option>' for i in range(len(results))))
        return template
    
if __name__ == '__main__':
    # 初始化图
    graph = PipelineGraph()

    # 添加节点
    graph.add_node('processor', {'type': 'transform', 'rate': 0.8})
    graph.add_node('validator', {'threshold': 0.5})

    # 添加边
    graph.add_edge('input', 'processor', 1)
    graph.add_edge('processor', 'validator', -1)
    graph.add_edge('validator', 'output', 0)

    # 更新参数
    graph.update_node_params('processor', {'rate': 1.0})

    graph.visualize() 
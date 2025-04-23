from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import openai
import sys
import os
from methods.self_RAG_demo import pipeline, graph
from citekit.utils.utils import parse_html_config

app = Flask(__name__)
CORS(app)  # 允许跨域请求


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    data = request.json  
    if not data:
        return jsonify({"error": "Invalid input data"}), 400
    
    try:
        result = pipeline(data)  # 直接调用 pipeline 处理数据
        print(result)
        return jsonify(result)  # 返回 JSON 结果
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/get_nodes", methods=["POST"])
def get_nodes(*args, **kwargs):
    graph.update()
    try:
        return jsonify(graph.get_json()) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update", methods=["POST"])
def update():

    data = request.json
    update_info = data.get("update_info")
    update_object = data.get('update_object')
    print(update_info, update_object)
    try:
        config, update_info = parse_html_config(update_info)
        print('GOT CONFIG', config, update_info)
        pipeline.update(update_object, config, update_info)
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/get_config", methods=["POST"])
def get_config():
    data = request.json
    config = data.get("config").lower()
    module_name = data.get("module_name")
    module = pipeline.get_module_by_name(module_name)

    try:
        if config in ['prompt', 'destination', 'max turn', 'global prompt', 'parallel']:
            return jsonify(module.get_json_config(config))
        else:
            raise NotImplementedError


    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    api_key = data.get("api_key")
    user_message = data.get("message")

    if not api_key or not user_message:
        return jsonify({"error": "API Key and message are required"}), 400

    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are a helpful assistant that follows the instructions of the user. You will be given a pipeline and (maybe) some datapoints in json format. You will be asked questions about the pipeline or the datapoints. Refuse to answer questions that are not about the pipeline or the datapoints."},
            {"role": "user", "content": user_message}
            ],
            stream=True  # 启用流式输出
        )

        def generate():
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    yield chunk["choices"][0]["delta"].get("content", "")


        return Response(generate(), content_type="text/event-stream")  # 使用流式响应
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import logging
import sys
from app import *
from config import *

app = Flask(__name__)
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/api/embed', methods=['GET'])
def createEmbed():
    if request.method == "GET":
        load_index()
    return jsonify({'status':'OK','message':'Created embeddings successfully!'}),200



@app.route('/api/upload', methods=['POST'])
def upload():
    if request.method == "POST":
        files = request.files.getlist("file")
        for file in files:
            file.save(os.path.join(f"{UPLOAD_FILE_PATH}", file.filename))
    return jsonify({'status':'OK','message':'File(s) uploaded successfully!'}),200

@app.route('/api/question', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    question = json['question']
    user_id = json['user_id']
    logging.info("post question `%s` for user `%s`", question, user_id)

    resp = chat(question, user_id)
    data = jsonify({'answer':resp})

    return data, 200

if __name__ == '__main__':
    init_llm()
    init_index()
    init_query_engine()

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
from flask import Flask, request, jsonify
import sys
sys.path.append('./newsgroups/')

from get_windows import lda
from lda2vec import lda2vec
app = Flask(__name__)

@app.route("/lda", methods=["POST"])
def getLda():
    path= request.json['path']
    return jsonify(lda(path))

@app.route("/lda2vec")
def getLda2vec():
    return jsonify(lda2vec())



if __name__ == '__main__':
    app.run(debug=True)

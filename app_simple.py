from flask import Flask
app = Flask(__name__)
app.run(host="127.0.0.1", port=5005, debug=True)
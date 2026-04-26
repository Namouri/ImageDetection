from flask import Flask, jsonify, render_template
from inference import classify_all_arms

app = Flask(__name__)

IMAGE_FOLDER = "camera2_images"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    results = classify_all_arms(IMAGE_FOLDER)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
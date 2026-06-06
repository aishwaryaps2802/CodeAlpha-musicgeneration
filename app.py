# app.py — Flask web interface

import os
from flask import Flask, render_template, send_file, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    try:
        from generate_music import generate
        generate(num_notes=200)
        return jsonify({
            'status':  'success',
            'message': 'Music generated successfully!'
        })
    except Exception as e:
        return jsonify({
            'status':  'error',
            'message': str(e)
        }), 500

@app.route('/download')
def download():
    path = 'output/generated.mid'
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'No music generated yet'}), 404

if __name__ == '__main__':
    app.run(debug=True)
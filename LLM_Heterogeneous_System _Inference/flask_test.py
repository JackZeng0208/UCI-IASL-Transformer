from flask import Flask, request
app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_connection():
    client_ip = request.remote_addr
    return f"Connection successful! Your IP is {client_ip}"

if __name__ == '__main__':
    # Allow the server to be accessible over the network
    app.run(host='0.0.0.0', port=5000, debug=True)

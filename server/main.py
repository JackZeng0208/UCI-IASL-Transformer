import json
from flask import Flask, request, jsonify
from model import build_opt_model

app = Flask(__name__)


@app.route("/decode", methods=["POST"])
def refine_text():
    data = request.get_json()
    ouput = model.generate(
        
    )

    
def main(
    model_size: str,
    ip: str,
    port: int,
):
    global model, tokenizer, config
    model, tokenizer, config = build_opt_model(model_size)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--model_size", type=str)
    parser.add_argument("--ip", type=str, help="specify IP address to run server", default="0.0.0.0")
    parser.add_argument("--port", type=int, help="specify port number to run server")
    args = parser.parse_args()
    
    main(
        model_size=args.model_size,
        ip=args.ip,
        port=args.port,
    )
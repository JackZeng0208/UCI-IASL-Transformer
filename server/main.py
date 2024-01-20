import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flask import Flask, request, jsonify
from model import build_model

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def welcome():
    return "Running server!"

@app.route("/decode", methods=["POST"])
def refine_text():
    data = request.get_json()
    print(f"received data: {data}")
    input_ids = torch.LongTensor(data.get("input_tokens")).to(device)
    draft_tokens = torch.LongTensor(data.get("draft_tokens")).to(device)
    top_k = data.get("top_k")
    top_p = data.get("top_p")

    model.eval()
    with torch.no_grad():
        target_outputs = dict(model.generate(
            torch.cat([input_ids, draft_tokens]).view(1, input_ids.shape[0]+draft_tokens.shape[0]),
            max_new_tokens=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=top_k,
            top_p=top_p
        ))
    print(f"Packing tensor output to serializable format...")
    # print(list(target_outputs.keys()))
    serializable_outputs = {}
    serializable_outputs["scores"] = [target_outputs["scores"][0][0].tolist()]
    serializable_outputs["sequences"] = target_outputs["sequences"][0].tolist()
    # for k, v in serializable_outputs.items():
    #     print(f"{k} -> {type(v)}, {len(v)}")
    print(f"Finish packing and send back generated outputs")
    return jsonify({ "target_outputs": serializable_outputs })

def main(
    model_size: str,
    host: str,
    port: int,
):
    global model, tokenizer, config, device
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = build_model(model_size)
    model.to(device)
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--model_size", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="specify IP address to run server")
    parser.add_argument("--port", type=int, default=5000, help="specify port number to run server")
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    main(
        model_size=args.model_size,
        host=args.host,
        port=args.port,
    )
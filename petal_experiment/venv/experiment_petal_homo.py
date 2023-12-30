from utils import model_evaluation
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigscience/bloom-560m"
# Load model directly


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
model = model.cuda()
model_evaluation(model,tokenizer,generate_token_num=100)
model_evaluation(model,tokenizer,generate_token_num=50)
model_evaluation(model,tokenizer,generate_token_num=10)
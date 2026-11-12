from flask import Flask, request;
from flask_cors import CORS;

from check_model_exists import checkModelExists;
from download_model import downloadModel;
from run_model import ModelPipeline;
from quantize_model import quantizeModel;

app = Flask(__name__);
CORS(app);

def templateResponse(success: bool = False, msg: str = "", details: dict = {}):
	return {
		"msg": msg,
		"success": success,
		"details": details
	};

if not checkModelExists():
	downloadModel();
	quantizeModel();

# mp: ModelPipeline = ModelPipeline();
mp_q: ModelPipeline = ModelPipeline(quantized = True);

@app.route("/health", methods=["GET"])
def health():
	return templateResponse(True, "The server is working.");

@app.route("/run_model", methods=["POST"])
def runModel():
	input_text: str = request.form.get("input_text");
#	quantized: bool = request.form.get("quantized");
#	if quantized == "0": quantized = False;
#	else: quantized = True;
#	if quantized: return mp_q.runModel(input_text);
#	else: return mp.runModel(input_text);
	return mp_q.runModel(input_text);


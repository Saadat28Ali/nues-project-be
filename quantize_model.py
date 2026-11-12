import os;
import torch;
from transformers import AutoConfig, AutoModelForSequenceClassification;

QUANTIZED_FILE = "bert_int8.pt";

def setEngine():
	# fbgemm = x86 CPUs, qnnpack = ARM CPUs (Apple Silicon, Raspberry Pi, etc.)
	engines = torch.backends.quantized.supported_engines;
	if "fbgemm" in engines:
		torch.backends.quantized.engine = "fbgemm";
	elif "qnnpack" in engines:
		torch.backends.quantized.engine = "qnnpack";


def applyQuantization(model):
	setEngine();
	return torch.ao.quantization.quantize_dynamic(
		model, {torch.nn.Linear}, dtype=torch.qint8
	);


def quantizeModel(model_path: str = "./model"):
	model = AutoModelForSequenceClassification.from_pretrained(
		model_path, local_files_only=True
	).eval();
	qmodel = applyQuantization(model);

	out_path = os.path.join(model_path, QUANTIZED_FILE);
	torch.save(qmodel.state_dict(), out_path);

	fp32_path = os.path.join(model_path, "pytorch_model.bin");
	if os.path.isfile(fp32_path):
		print(f"FP32: {os.path.getsize(fp32_path) / 1e6:.0f} MB");
	print(f"INT8: {os.path.getsize(out_path) / 1e6:.0f} MB -> {out_path}");


def loadQuantizedModel(model_path: str = "./model"):
	weights_path = os.path.join(model_path, QUANTIZED_FILE);
	if not os.path.isfile(weights_path):
		raise FileNotFoundError(
			f"{weights_path} not found. Run `python quantize_model.py` first."
		);

	# Rebuild the architecture (no weights), quantize it, then load the INT8 weights
	config = AutoConfig.from_pretrained(model_path, local_files_only=True);
	model = AutoModelForSequenceClassification.from_config(config).eval();
	model = applyQuantization(model);
	model.load_state_dict(torch.load(weights_path));
	return model;


if __name__ == "__main__":
	quantizeModel();

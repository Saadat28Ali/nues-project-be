from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification;

class ModelPipeline:
	def __init__(self, model_path: str = "./model", quantized: bool = False):
		self.pipe = None;
		self.tokenizer = None;
		self.model = None;
		self.__createPipeline(model_path, quantized);

	def __createPipeline(self, model_path, quantized):
		self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True);
		if quantized:
			# Imported here so the original (FP32) path doesn't need the quantization module
			from quantize_model import loadQuantizedModel;
			self.model = loadQuantizedModel(model_path);
		else:
			self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True);
		# Dynamic INT8 models are CPU-only, so pin the pipeline to CPU in both modes
		self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=-1);

	def runModel(self, input_text: str):
		return self.pipe(input_text);

USE_QUANTIZED = True;  # True = INT8 quantized model, False = original FP32 model

if __name__ == "__main__":
	mp: ModelPipeline = ModelPipeline(quantized=USE_QUANTIZED);
	print(f"Using {'INT8 quantized' if USE_QUANTIZED else 'original FP32'} model");
	while True:
		i: str = input(">> ");
		print(mp.pipe(i));
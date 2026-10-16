from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification;

class ModelPipeline:
	def __init__(self, model_path: str ="./model"):
		self.pipe = None;
		self.tokenizer = None;
		self.model = None;
		self.__createPipeline(model_path);

	def __createPipeline(self, model_path):
		self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True);
		self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True);
		self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer);

	def runModel(self, input_text: str):
		return self.pipe(input_text);

if __name__ == "__main__":
	mp: ModelPipeline = ModelPipeline();
	while True:
		i: str = input(">> ");
		print(mp.pipe(i));

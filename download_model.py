from huggingface_hub import snapshot_download;

def downloadModel():
	snapshot_download(
		"LiYuan/amazon-review-sentiment-analysis",
		local_dir="./model",
		allow_patterns=["*.json", "pytorch_model.bin", "*.txt"]
	);

if __name__ == "__main__":
	pass;

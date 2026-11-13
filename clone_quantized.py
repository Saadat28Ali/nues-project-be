from huggingface_hub import snapshot_download

def cloneQuantized():
	snapshot_download("Saadat28Ali/nues", local_dir="./model")

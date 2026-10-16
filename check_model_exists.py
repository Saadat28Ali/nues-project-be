from os import path, getcwd

ROOT_DIR: str = path.abspath(getcwd());

def checkModelExists() -> bool:
	return path.exists(path.abspath(ROOT_DIR + "/model"));

if __name__ == "__main__":
	print(checkModelExists());

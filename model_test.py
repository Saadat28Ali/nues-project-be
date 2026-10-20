import pandas as pd;
from os import path, getcwd;
from sys import argv;
from math import ceil;

from progress.bar import IncrementalBar;
from progress.spinner import Spinner;

from run_model import ModelPipeline;

ROOT_DIR: str = path.abspath(getcwd());

def getTestDataset(filepath: str, head_count: int = 5) -> pd.DataFrame:
	# Returns a subset of the dataset, containing only
	# the columns containing review test and the score
	# --------------------------------------------------
	# The subset dataset is generated from the complete
	# test dataset, by first grouping by score and then
	# sampling ceil(head_count / 5) rows from each
	# group. This way the subset dataset will always
	# have data that is approximately balanced in terms
	# of the score, i.e., the number of rows that have
	# a score of 1, and the number of rows that have a
	# score of say, 5, will be almost equal
	# --------------------------------------------------

	test_data: pd.DataFrame = pd.read_csv(filepath) \
								.set_index("Id") \
								.groupby("Score") \
								.sample(n=ceil(head_count / 5), random_state=42) \
								.sample(frac=1, random_state=42) \
								.head(head_count) \
								.sort_index();

	test_data_filt_cols: pd.DataFrame = test_data[["Text", "Score"]];
	for row in test_data_filt_cols.itertuples():
		test_data_filt_cols.at[row.Index, "Score"] = int(row.Score);
	return test_data_filt_cols;

def processModelResponse(response: list[dict]) -> dict:
	# Processes return value from the model
	# Converts a list of dictionary to another
	# dictionary, but with different keys
	# --------------------------------------------------

	return {
		"score": int(response[0]["label"][0]),
		"confidence": round(float(response[0]["score"]), 4)
	};

def performTest(test_data: pd.DataFrame, mp: ModelPipeline, head_length: int = 5) -> None:
	# Passed test data to the model and attaches the
	# results to the test data data frame
	# --------------------------------------------------

	with IncrementalBar("Performing test", max=head_length) as bar:
		for row in test_data.head(head_length).itertuples():
			result: dict = processModelResponse(mp.pipe(row.Text[:512:]));
			test_data.at[row.Index, "PredictedScore"] = result["score"];
			test_data.at[row.Index, "Confidence"] = result["confidence"];
			test_data.at[row.Index, "ScoreDiff"] = int(result["score"] - row.Score);
			test_data.at[row.Index, "AbsScoreDiff"] = abs(int(result["score"] - row.Score));
			test_data.at[row.Index, "LargeAbsDiff"] = abs(int(result["score"] - row.Score)) >= 2;
			bar.next();
def main():
	HEAD_LENGTH: int = int(argv[1]);
	print("Creating subset dataset...");
	test_data: pd.DataFrame = getTestDataset(path.abspath(ROOT_DIR + "/datasets/test/reviews.csv"), HEAD_LENGTH);
	print("Loading model...");
	mp: ModelPipeline = ModelPipeline(path.abspath(ROOT_DIR + "/model"));
	performTest(test_data, mp, HEAD_LENGTH);
	print("Writing results to file...");
	test_data.to_csv(path.abspath(ROOT_DIR + "/datasets/test_results.csv"));
#	for row in test_data.head(HEAD_COUNT).itertuples():
#		if row.PredictedScore not in {1, 2, 3, 4, 5}:
#			print(row);
#			print(f"Retry result: {mp.pipe(row.Text)}")
##			break;

if __name__ == "__main__":
	main();

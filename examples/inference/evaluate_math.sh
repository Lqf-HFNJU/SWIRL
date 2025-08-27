PROJECT_DIR="$(pwd)"

inference_path=$PROJECT_DIR/eval/SWIRL_MATH_data/test/MATH500_test.parquet # path to inference results (.parquet)
save_json_path=$PROJECT_DIR/eval/SWIRL_MATH_data/test/MATH500_test.json # path to save evaluation results


python -m multi_agent.evals.evaluation_math --fp $inference_path --out $save_json_path
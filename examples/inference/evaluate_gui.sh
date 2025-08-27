PROJECT_DIR="$(pwd)"

inference_path=$PROJECT_DIR/eval/SWIRL_GUI_data/test/high_level/AndroidControl_test.parquet # path to inference results (.parquet)
save_json_path=$PROJECT_DIR/eval/SWIRL_GUI_data/test/high_level/AndroidControl_test.json # path to save evaluation results


python -m multi_agent.evals.evaluation_gui --fp $inference_path --out $save_json_path
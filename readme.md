docker run --rm -p 8501:8501 --init -v D:/Development/personal/AIMapper/model:/models/test/1 -e MODEL_NAME="test" tensorflow/serving
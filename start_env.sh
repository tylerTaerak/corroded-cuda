docker build . -t rscuda-dev --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)

docker run -it -v .:/app -u dev_user -w /app --runtime=nvidia --gpus=all rscuda-dev bash

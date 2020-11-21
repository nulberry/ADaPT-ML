# CECN Data Programming

Repo for creating training data through heuristics and external knowledge, as well as manually annotating small samples of data. Right now, we are using the Personal Values Dictionary developed by Ponizovskiy et al. (2020) to automatically create training data for
modelling the personal values people are expressing in text.

![Data programming and classification flow.](/graphics/dp_class_flow.png "dp and class flow")

https://towardsdatascience.com/targeted-sentiment-analysis-vs-traditional-sentiment-analysis-4d9f2c12a476
https://github.com/songyouwei/ABSA-PyTorch
https://hub.docker.com/_/mysql?tab=description

## Setup

```shell script
cd $DP_REPO_PATH
docker-compose up -d --build
```
*Note:* make sure that you export values for the environment variables in docker-compose.yml

Will create and start the MySQL backend store, the MLFlow server, and the
interface for running MLFlow experiments.

```shell script
docker attach dp-mlflow-server
wait-for-it dp-mlflow-db:3306 -s -- mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root ./mlruns --host 0.0.0.0
```
or
```shell script
docker exec dp-mlflow-server 'wait-for-it dp-mlflow-db:3306 -s -- mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root ./mlruns --host 0.0.0.0'
```
Will start the MLFlow server manually

## Main Usage

```shell script
docker attach dp-mlflow
wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e <pv> --experiment-name <energyeast> -P data=/unlabeled_data</path/to/data.pkl> .
```
or
```shell script
docker exec dp-mlflow 'wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e <entrypoint> --experiment-name <name> -P data=/unlabeled_data</path/to/data.pkl> .'
```
*-e pv*: the entrypoint is for assessing personal values

*-P pv_data*: path to the unlabeled training data

Go to http://129.128.215.241:5000 to view the experiments

## Labeling data with Label Studio

```shell script
docker run \
-p 8080:8080 \
-v $LS_PATH:/label-studio/session_projects \
-v $LS_PATH/pv_config.xml:/pv_config.xml \
-v $LS_PATH/pv_tasks.json:/pv_tasks.json \
--name label-studio \
heartexlabs/label-studio:latest \
label-studio start-multi-session \
--root-dir ./session_projects \
--force \
--label-config /pv_config.xml \
--input-path /pv_tasks.json \
--initial-project-description personal_values
```

This will import the JSON-formatted list of data points in each file in the input path. The files should look like this:
```json
[
  {
    "tweet_text": "Opossum is great",
    "ref_id": "<tweet_id>",
    "meta_info": {
      "timestamp": "2020-03-09 18:15:28.212882",
      "location": "North Pole"
    }
  }
]
```

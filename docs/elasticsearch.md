# Running Elasticsearch locally on MacOS using Docker

## Install Colima

```
brew install colima
colima start -m 4
```

[Colima](https://github.com/abiosoft/colima) is a Docker runtime with good
support for Mac M1s.

Unfortunately, you will need to give the JVM 4 gigs of RAM in order to set the
password in the next step. Probably you can stop and restart Colima with less
RAM after the setup is done.

## Install and run the Elasticsearch container

```
docker pull elasticsearch:8.5.3
docker volume create --name elasticsearch
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -v elasticsearch:/usr/share/elasticsearch/data \
  elasticsearch:8.5.3
```

## Configure the Elasticsearch credentials

```
docker exec -it elasticsearch bin/elasticsearch-reset-password -a -u elastic
```

## Test the Elasticsearch credentials

```
curl -k -u elastic:<password> https://localhost:9200/
```

The `-k` option to `curl` tells it not to worry about the self-signed cert. You
should get a JSON doc back indicating success.

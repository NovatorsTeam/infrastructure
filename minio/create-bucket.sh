#!/bin/bash

create_bucket_cmdname=${0##*/}

usage() 
{
    cat << USAGE >&2
Usage:
    $create_bucket_cmdname [--host] [--user] [--password]
    --host=http://host:port         Minio server for bucket creation
    --user=MINIO_USER               Minio user for bucket creation
    --password=MINIO_PASSWORD       Minio password for bucket creation
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi
  shift
done

if [[ -z $host ]]; then
    usage
    echo "Missing parameter --host"
    exit 1
elif [[ -z $user ]]; then
    usage
    echo "Missing parameter --user"
    exit 1
elif [[ -z $password ]]; then
    usage
    echo "Missing parameter --password"
    exit 1
fi

# https://github.com/minio/mc
mc alias set minioserver $host $user $password
mc mb minioserver/mlflow
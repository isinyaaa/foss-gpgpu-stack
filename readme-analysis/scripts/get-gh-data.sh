#!/usr/bin/env bash

set -o pipefail

DATA_DIR="data/github"

REPO_FLAGS="--stars '>=5' --followers '>=5' --archived=false --visibility 'public' --language 'glsl' --language 'c' --language 'c++' --updated '>=2021-01-01' --limit 1000 --sort 'stars'"
REPO_OUTPUT="--json fullName,forksCount,stargazersCount,openIssuesCount,updatedAt,description,language"

ISSUE_FLAGS="--sort 'updated' --order 'desc' --limit 1000"
ISSUE_OUTPUT="--json createdAt,updatedAt,commentsCount,state"

cd "$DATA_DIR" || exit 1

for api in {cuda,opencl,opengl,vulkan}; do
    echo "Getting $api data"
    eval gh search repos "$api" "$REPO_FLAGS" "$REPO_OUTPUT" | jq | tee "${api}_output".json
    cat "${api}_output".json | jq 'map(.fullName)' | sed -e 's/"//g' -e 's/,//g' -e '1d;$d' | tee "${api}_repos".txt
    while read -r repo; do
        echo "Getting $repo issues"
        repo_name=$(echo "$repo" | sed 's/\//_/g')
        filename="${api}/${repo_name}".json
        echo '[' > "$filename"
        while true; do
            eval gh search issues --repo "$repo" "$ISSUE_FLAGS" --state 'open' "$ISSUE_OUTPUT"  | jq | tee -a 
            if [ $? -ne 0 ]; then
                echo "Failed to get issues for $repo"
                sleep 5
            else
                break
            fi
        done
        echo ',' > "$filename"
        sleep 1
        while true; do
            eval gh search issues --repo "$repo" "$ISSUE_FLAGS" --state 'closed' "$ISSUE_OUTPUT" | jq | tee -a "${api}/${repo_name}".json
            if [ $? -ne 0 ]; then
                echo "Failed to get issues for $repo"
                sleep 5
            else
                break
            fi
        done
        sleep 1
        echo ']' >> "$filename"
        tmp=$(mktemp)
        jq -s 'flatten' "$filename" > "$tmp" && mv "$tmp" "$filename"
    done < "${api}_repos".txt
done

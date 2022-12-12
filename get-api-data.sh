#!/usr/bin/env bash

set -o pipefail

REPO_FLAGS="--stars '>=5' --followers '>=5' --archived=false --visibility 'public' --language 'glsl' --language 'c' --language 'c++' --updated '>=2021-01-01' --limit 1000 --sort 'stars'"
REPO_OUTPUT="--json fullName,forksCount,stargazersCount,openIssuesCount,updatedAt,description,language"

ISSUE_FLAGS="--sort 'updated' --order 'desc' --limit 1000"
ISSUE_OUTPUT="--json createdAt,updatedAt,commentsCount,state"

for api in {cuda,opencl,opengl,vulkan}; do
    echo "Getting $api data"
    eval gh search repos "$api" "$REPO_FLAGS" "$REPO_OUTPUT" | jq | tee "gh_data/${api}_output".json
    cat "gh_data/${api}_output".json | jq 'map(.fullName)' | sed -e 's/"//g' -e 's/,//g' -e '1d;$d' | tee "gh_data/${api}_repos".txt
    while read -r repo; do
        echo "Getting $repo issues"
        repo_name=$(echo "$repo" | sed 's/\//_/g')
        filename="gh_data/${api}/${repo_name}".json
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
            eval gh search issues --repo "$repo" "$ISSUE_FLAGS" --state 'closed' "$ISSUE_OUTPUT" | jq | tee -a "gh_data/${api}/${repo_name}".json
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
    done < "gh_data/${api}_repos".txt
done

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
        issue_query_out=$(mktemp)
        for state in open closed; do
            slept=0
            tmp=$(mktemp)
            while [ $slept -lt 600 ]; do
                eval gh search issues --repo "$remote" "$ISSUE_FLAGS" --state "$state" "$ISSUE_OUTPUT" | jq > "$tmp"
                if [ $? -ne 0 ]; then
                    echo "Failed to get issues for $remote"
                    sleep 5
                    slept=$((slept + 5))
                else
                    {
                        echo '['
                        cat "$tmp"
                        echo '],'
                    } >> "$issue_query_out"
                    break
                fi
            done
        done
        # fallback for an empty list or a trailing comma
        echo '[]' >> "$issue_query_out"
        jq -s 'flatten' "$issue_query_out" > "${api}/${repo//\//_}.json"
    done < "${api}_repos".txt
done

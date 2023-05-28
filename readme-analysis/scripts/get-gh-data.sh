#!/usr/bin/env bash

DATA_DIR="data/github"

REPO_FLAGS=(
    --stars '>=5'
    --followers '>=5'
    --archived=false
    --visibility 'public'
    --language 'glsl'
    --language 'c'
    --language 'c++'
    --updated '>=2021-01-01'
    --limit 1000
    --sort 'stars'
)
REPO_OUTPUT=(
    fullName
    forksCount
    stargazersCount
    openIssuesCount
    updatedAt
    description
    language
)
repo_output=$(IFS=,; echo "${REPO_OUTPUT[*]}")

ISSUE_FLAGS=(
    --sort 'updated'
    --order 'desc'
    --limit 1000
)
ISSUE_OUTPUT=(
    createdAt
    updatedAt
    commentsCount
    state
)
issue_output=$(IFS=,; echo "${ISSUE_OUTPUT[*]}")

cd "$DATA_DIR" || exit 1

for api in {cuda,opencl,opengl,vulkan}; do
    echo "Getting $api data"
    repo_query_out="${api}.json"

    (
        set -o pipefail
        gh search repos $api "${REPO_FLAGS[@]}" --json "$repo_output" | jq > "$repo_query_out"
    )

    repo_list="${api}_repos.txt"
    # we map the fullName from each entry to a list
    # then we remove the first and last lines (the brackets), quotes and commas
    jq 'map(.fullName)' "$repo_query_out" |\
        sed -e 's/"//g' -e 's/,//g' -e '1d;$d' > "$repo_list"

    while read -r repo; do
        echo "Getting $repo issues"
        issue_query_out=$(mktemp)
        for state in open closed; do
            slept=0
            tmp=$(mktemp)
            while [ $slept -lt 600 ]; do
                if gh search issues --repo "$repo" "${ISSUE_FLAGS[@]}" --state $state --json "$issue_output" > "$tmp"; then
                    echo "Failed to get issues for $repo"
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
    done < "$repo_list"
done

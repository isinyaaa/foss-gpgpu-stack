#!/usr/bin/env python3

import datetime
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = 'data/github'


def get_repo_data(filename):
    with open(os.path.join('.', filename)) as f:
        data = json.load(f)

    api_name = re.sub(r'_output\.json$', '', filename)

    df_data = pd.DataFrame(data).assign(api=lambda _: api_name).\
                    rename(columns={'stargazersCount': 'stars'}).\
                    rename(columns={'forksCount': 'forks'}).\
                    rename(columns={'updatedAt': 'last_updated'}).\
                    rename(columns={'fullName': 'project'}).\
                    iloc[:, [1, 4, 0, 2, 3]].\
                    assign(closed_issues=lambda _: 0).\
                    assign(open_issues=lambda _: 0).\
                    assign(engagement=lambda _: 0)

    return df_data


def get_issues_data(path):
    with os.scandir(path) as it:
        for entry in it:
            with open(os.path.join(path, entry.name)) as f:
                data = json.load(f)

            df_data = pd.DataFrame(data).\
                    rename(columns={'commentsCount': 'comments'}).\
                    rename(columns={'createdAt': 'created'}).\
                    rename(columns={'updatedAt': 'last_updated'}).\
                    assign(delta_t=lambda _: 0)

            df_data['created'] = pd.to_datetime(df_data['created'], utc=True)
            df_data['last_updated'] = pd.to_datetime(df_data['last_updated'], utc=True)
            df_data['delta_t'] = df_data['last_updated'] - df_data['created']

    return df_data


def main():
    api_data = pd.DataFrame()
    with os.scandir('.') as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.json'):
                repo_data = get_repo_data(entry.name)
                api_data = pd.concat([api_data, repo_data])
                continue
            elif entry.is_dir():
                path = os.path.join('.', entry.name)
                issues_data = get_issues_data(path)

    # print distribution of forks and stars for each API
    # colors = ['#009e73', '#1f77b4', '#ff7f0e', '#e62323']
    # sns.set_palette(sns.color_palette(colors))
    # sns.set_style('whitegrid')
    # sns.set_context('paper')
    # sns.set(font_scale=1.5)
    # sns.set(rc={'figure.figsize':(11.7,8.27)})

    # sns.boxplot(data=api_data, x='api', y='stars', showfliers=False)
    # plt.title('Distribution of Stars per GAPI')
    # plt.savefig('plots/stars_boxplot.png', dpi=300)
    # plt.clf()

    # sns.boxplot(data=api_data, x='api', y='forks', showfliers=False)
    # plt.title('Distribution of Forks per GAPI')
    # plt.savefig('plots/forks_boxplot.png', dpi=300)
    # plt.clf()

    # sns.histplot(data=api_data, x='stars', hue='api', bins=50, stat='density', common_norm=False)
    # plt.title('Distribution of Stars per GAPI')
    # plt.savefig('plots/stars_hist.png', dpi=300)
    # plt.clf()

    # sns.kdeplot(data=api_data, x='stars', hue='api', common_norm=False)
    # plt.title('Distribution of Stars per GAPI')
    # plt.savefig('plots/stars_kde.png', dpi=300)
    # plt.clf()

    # sns.histplot(data=api_data, x='stars', hue='api', bins=len(api_data),
    #              stat='density', common_norm=False, element='step', fill=False,
    #              cumulative=True)
    # plt.title('Distribution of Stars per GAPI')
    # plt.savefig('plots/stars_cumulative.png', dpi=300)
    # plt.clf()


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    os.chdir(DATA_DIR)

    main()

import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import ttest_ind, tukey_hsd
from functools import reduce
import dataframe_image as dfi

#Set table colour style
def colour_rows(row):
    if row['category'] == 'deployment_name':
        return ['background-color: lightblue'] * len(row)
    elif row['category'] == 'embedding_model':
        return ['background-color: lightgreen'] * len(row)
    elif row['category'] == 'search_algorithm':
        return ['background-color: lightyellow'] * len(row)
    elif row['category'] == 'agentic_rerank':
        return ['background-color: lightsalmon'] * len(row)
    else:
        return ['background-color: white'] * len(row)
        
def clean_comparison_label(label):
    #Cleans the group label for the output.
    if isinstance(label, tuple):
        label = ', '.join(str(x) for x in label)
    label = str(label)
    label = label.replace("np.", "")
    label = label.replace("_", "")
    label = label.replace("(", "").replace(")", "").replace(",", "")
    label = label.strip("'").strip('"')
    return label

def compare_groups(df, group_columns, metric, category):

    groups = df.groupby(group_columns)
    group_dfs = []
    group_names = []
    #Extract all the groups and their respective dataframes
    for group, dataframe in groups:
        group_names.append(clean_comparison_label(group))
        group_dfs.append(dataframe[metric])

    print(f"Comparing: " + ", ".join((str(x) for x in group_names)))

    if len(groups) == 2:
        #T-test for 2 groups
        t_stat, p_value = ttest_ind(*group_dfs)
        temp_dict = {
            "category": [category],
            "comp1": [group_names[0]],
            "comp2": [group_names[1]],
            metric: [p_value]
        }
        return pd.DataFrame(temp_dict)

    elif len(groups) == 3:
        #Tukey hsd test for 3 groups
        test_result = tukey_hsd(*group_dfs)
        results = []
        for x,y in ((0,1), (0,2), (1,2) ):
            temp_dict = {
                "category": category,
                "comp1": group_names[x],
                "comp2": group_names[y],
                metric: test_result.pvalue[x][y]
            }
            results.append(temp_dict)
        return pd.DataFrame(results)
    else:
        raise Exception("Number of groups compared not 2 or 3")

def compare_dataframe(df, column_groups, metrics, categories):
    #Define
    all_metrics = []
    #Loop over all metrics to test.
    for metric in metrics:
        results = []
        for category, columns in zip(categories, column_groups):
            #Loop over all results to test.
            results.append(compare_groups(df,columns, metric, category))

        #Make one dataframe from all the results and add them to all the metric results.
        all_metrics.append(pd.concat(results))
        
    #Merge all dataframes for each metric into one dataframe.
    temp_df = reduce(lambda left, right: pd.merge(left, right, on=['category', 'comp1', 'comp2']), all_metrics)
    return temp_df

def ragas_main():
    ragas_statistics_df = pd.read_pickle("data/ragas_statistics.pkl")
    group_columns = [['deployment_name'] ,['embedding_model'], ['search_algorithm'], ['agentic_retrieval', 'reranking']]
    metrics = ['faithfulness', 'context_recall','context_precision', 'aggregated_score']
    categories = ['deployment_name', 'embedding_model', 'search_algorithm', 'agentic_rerank']
    ragas_df = compare_dataframe(ragas_statistics_df,
                    group_columns,
                    metrics,
                    categories)

    #Style the final dataframe.
    ragas_df = ragas_df.style.apply(colour_rows, axis=1)
    #Export dataframe as image.
    dfi.export(ragas_df, "images/ragas.png")

def rouge_main():
    rouge_statistics_df = pd.read_pickle("data/rouge_statistics.pkl")
    group_columns = [['deployment_name'] ,['embedding_model'], ['search_algorithm'], ['agentic_retrieval', 'reranking']]
    metrics = ['rouge1_precision', 'rouge1_recall',
       'rouge1_fmeasure', 'rougeL_precision', 'rougeL_recall',
       'rougeL_fmeasure']
    categories = ['deployment_name', 'embedding_model', 'search_algorithm', 'agentic_rerank']
    rouge_df = compare_dataframe(rouge_statistics_df,
                                group_columns,
                                metrics,
                                categories)

    #Style the final dataframe.
    rouge_df = rouge_df.style.apply(colour_rows, axis=1)
    #Export dataframe as image.
    dfi.export(rouge_df, "images/rouge.png")
ragas_main()
rouge_main()
import seaborn as sns
import os
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from utils.eval_helpers import ConfusionMatrix, cm_dict_to_subjective

_baseline='#236AA9'
_ours='#236AA9'
_oracle='#236AA9'
_err1='#CE8950'
_err2='#548C67'
_err3='#D7C270'
_err4='#9E6B6D'
_extra='#B4B36A'

method_color_list = [_baseline, _baseline, _ours, _ours, _oracle]
method_marker_list = ['x','o','^','^','^','s']
color_list = ['#FC8D62','#66C2A5','#E78AC3','#8DA0CB']
color_list_light = ['#FEC5B0','#B2E0D2','#F3C5E1','#C5D0E5']
hatches = ['|','O','x','.']
method_color_list_lines_dict = {
    'GPT':_baseline,
    'rules':_err2,
    'ours_with_no_concepts':_extra,
    'ours_with_gpt':_err4,
    'ours_with_model':_err1,
    'ours_with_user':_err3
}

method_color_list_lines = [_err2, _baseline, _extra, _err4, _err1, _err3]


def embedding_space_visualizer(viz_results, output_file):
    """
    Input: dictionary
        'item_embeddings':torch.tensor() of embeddings, 
        'concept_embeddings':torch.tensor() of embeddings, 
        'concepts_categories':index determining which color to plot as
        'concepts_labels': names to use in legend for first object of each color

    This function combines all item and concept embeddings, normalizes them, and geenrates  tsne visualization of the embeddings. 
    Each embedding should be colored by the corresponding concept category, and one item in each category should be labeled with the category name.
    The item embeddings should be plotted with the '.' marker, and the concept embeddings should be plotted with the 'x' marker.
    """

    item_embeddings = torch.nn.functional.normalize(viz_results['item_embeddings'].detach().clone(), dim=-1)
    concept_embeddings = torch.nn.functional.normalize(viz_results['concept_embeddings'].detach().clone(), dim=-1)
    concepts_categories = deepcopy(viz_results['item_categories'] + viz_results['concepts_categories'])
    concepts_labels = deepcopy(viz_results['concepts_labels'])
    concepts_labels_left = deepcopy(viz_results['concepts_labels'])

    embeddings = torch.cat([item_embeddings, concept_embeddings], dim=0)
    embeddings = embeddings - embeddings.mean(dim=0)
    embeddings = embeddings / embeddings.std(dim=0)

    tsne = TSNE(n_components=2, random_state=0)
    embeddings = tsne.fit_transform(embeddings.detach().numpy())

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(len(embeddings)):
        if i < len(item_embeddings):
            marker = '.'
        else:
            marker = 'x'
        if concepts_categories[i] in concepts_labels_left:
            concepts_labels_left.remove(concepts_categories[i])
            print("\t\tlabel",concepts_categories[i])
            ax.scatter(embeddings[i,0], embeddings[i,1], marker=marker, color=color_list[concepts_labels.index(concepts_categories[i])], label=concepts_categories[i])
        else:
            ax.scatter(embeddings[i,0], embeddings[i,1], marker=marker, color=color_list[concepts_labels.index(concepts_categories[i])])
    ax.legend()
    plt.savefig(output_file)
    
def results_visualizer_assistance(logs_dir, persona, num_train_actions, prefix=''):
    combined_metrics = {
        'ours_with_no_concepts':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'ours_with_model':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'ours_with_gpt':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'ours_with_user':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'ours_with_model_latent':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'rules':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        'GPT':{'accuracy':0, 'f1_macro':0, 'mrr_output_based':0, 'mrr_gt_based':0, 'mrr_attn_based':0, 'mrr_correct':0, 'mrr_wrong':0, 'subj':{}},
        }
    
    our_method = json.load(open(os.path.join(logs_dir, 'Seen_{:02d}'.format(num_train_actions), persona, prefix+'assistances_final.json')))
    baseline_gpt_path = os.path.join(logs_dir, 'baseline_assist', persona, 'gpt', 'Seen_{:02d}'.format(num_train_actions), 'confusion_matrix.json')
    if os.path.exists(baseline_gpt_path):
        baseline_gpt = json.load(open(baseline_gpt_path))['final']
    else:
        baseline_gpt = ConfusionMatrix(['do_now', 'do_later', 'remind', 'no_action']).get_numbers_dict()
        baseline_gpt['mrr'] = 0
    baseline_rules_file = os.path.join(logs_dir, 'baseline_assist', persona, 'rules', 'Seen_{:02d}'.format(num_train_actions), 'confusion_matrix.json')
    if os.path.exists(baseline_rules_file):
        baseline_rules = json.load(open(baseline_rules_file))['final']
    else:
        baseline_rules = ConfusionMatrix(['do_now', 'do_later', 'remind', 'no_action']).get_numbers_dict()
        baseline_rules['mrr'] = 0
    our_method_with_gpt = our_method['with_gpt']['cf']
    our_method_with_model = our_method['with_model']['cf']
    our_method_with_no_concepts = our_method['with_no_concepts']['cf']
    our_method_with_model_latent = our_method['with_model_latent']['cf']
    our_method_with_user = our_method['with_user']['cf']

    combined_metrics['ours_with_no_concepts']['accuracy'] = (our_method_with_no_concepts['accuracy'])
    combined_metrics['ours_with_model']['accuracy'] = (our_method_with_model['accuracy'])
    combined_metrics['ours_with_gpt']['accuracy'] = (our_method_with_gpt['accuracy'])
    combined_metrics['ours_with_model_latent']['accuracy'] = (our_method_with_model_latent['accuracy'])
    combined_metrics['ours_with_user']['accuracy'] = (our_method_with_user['accuracy'])
    combined_metrics['rules']['accuracy'] = (baseline_rules['accuracy'])
    combined_metrics['GPT']['accuracy'] = (baseline_gpt['accuracy'])  
    
    combined_metrics['ours_with_no_concepts']['subj'] = cm_dict_to_subjective(our_method_with_no_concepts['confusion_matrix'])
    combined_metrics['ours_with_model']['subj'] = cm_dict_to_subjective(our_method_with_model['confusion_matrix'])
    combined_metrics['ours_with_gpt']['subj'] = cm_dict_to_subjective(our_method_with_gpt['confusion_matrix'])
    combined_metrics['ours_with_model_latent']['subj'] = cm_dict_to_subjective(our_method_with_model_latent['confusion_matrix'])
    combined_metrics['ours_with_user']['subj'] = cm_dict_to_subjective(our_method_with_user['confusion_matrix'])
    combined_metrics['rules']['subj'] = cm_dict_to_subjective(baseline_rules['confusion_matrix'])
    combined_metrics['GPT']['subj'] = cm_dict_to_subjective(baseline_gpt['confusion_matrix'])

    fig, axs = plt.subplots(3, 3, figsize=(30, 25))
    fig.suptitle(persona)
    ## Plot the accuracy comparison
    axs[0, 0].bar(['GPT', 'Rules', 'Direct Transformer', 'Our Method\n(GPT concepts)', 'Our Method\n(Model concepts)', 'Our Method\n(User Concepts)', 'Our Method\n(Model latent)'], 
                  [baseline_gpt['accuracy'], baseline_rules['accuracy'], our_method_with_no_concepts['accuracy'], our_method_with_gpt['accuracy'], our_method_with_model['accuracy'], our_method_with_user['accuracy'], our_method_with_model_latent['accuracy']])
    axs[0,0].set_title('Accuracy Comparison')
    
    f1_macros = []
    ## Plot the confusion matrices
    cm = ConfusionMatrix.from_dict(baseline_gpt['confusion_matrix'])
    classes = cm.classes
    sns.heatmap(cm.cm_array, annot=True, ax=axs[0, 1], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['GPT']['f1_macro'] = (cm.f1_macro())
    axs[0, 1].set_title('GPT')
    cm = ConfusionMatrix.from_dict(baseline_rules['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[0, 2], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['rules']['f1_macro'] = (cm.f1_macro())
    axs[0, 2].set_title('Rules')
    cm = ConfusionMatrix.from_dict(our_method_with_no_concepts['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[1, 1], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['ours_with_no_concepts']['f1_macro'] = (cm.f1_macro())
    axs[1, 1].set_title('Direct Transformer')
    cm = ConfusionMatrix.from_dict(our_method_with_gpt['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[1, 1], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['ours_with_gpt']['f1_macro'] = (cm.f1_macro())
    axs[1, 1].set_title('Our Method (GPT concepts)')
    cm = ConfusionMatrix.from_dict(our_method_with_model['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[1, 2], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['ours_with_model']['f1_macro'] = (cm.f1_macro())
    axs[1, 2].set_title('Our Method (Model concepts)')
    cm = ConfusionMatrix.from_dict(our_method_with_user['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[2, 1], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['ours_with_user']['f1_macro'] = (cm.f1_macro())
    axs[2, 1].set_title('Our Method (User Concepts)')
    cm = ConfusionMatrix.from_dict(our_method_with_model_latent['confusion_matrix'], classes)
    sns.heatmap(cm.cm_array, annot=True, ax=axs[2, 2], fmt='g')
    f1_macros.append(cm.f1_macro())
    combined_metrics['ours_with_model_latent']['f1_macro'] = (cm.f1_macro())
    axs[2, 2].set_title('Our Method (Model latents)')
    
    for ax in [axs[0,1], axs[0,2], axs[1,1], axs[1,2], axs[2,1], axs[2,2]]:
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
    
    # Use last subplot for F1 Score comparison
    axs[1, 0].bar(['GPT', 'Rules', 'Direct Transformer', 'Our Method\n(GPT conepts)', 'Our Method\n(Model concepts)', 'Our Method\n(User concepts)', 'Our Method\n(Model latent)'], f1_macros)
    axs[1, 0].set_title('F-1 Score Comparison')
    
    ## Use last plot for MRR comparison
    labels = []
    values = []
    names = {
        'with_gpt': 'GPT Conc.',
        'with_no_concepts': 'Direct Transformer',
        'with_model': 'Model Conc.',
        'with_user': 'User Conc.',
        'with_model_latent': 'Model Latent',
        'output_based': 'Output Expl.',
        'output_based_correct': 'Output Expl. Correct',
        'output_based_wrong': 'Output Expl. Wrong',
        'gt_based': 'GT Expl.',
        'gt_based_correct': 'GT Expl. Correct',
        'gt_based_wrong': 'GT Expl. Wrong',
        'attn_based': 'Attn Expl.',
        'attn_based_correct': 'Attn Expl. Correct',
        'attn_based_wrong': 'Attn Expl. Wrong'
    }
    for i,key_method in enumerate(our_method.keys()):
        if key_method == 'left_to_do': continue
        for key_src in our_method[key_method]['MRR'].keys():
            labels.append(f'{names[key_method]}\n{names[key_src]}')
            values.append(our_method[key_method]['MRR'][key_src])
            combined_metrics[f'ours_{key_method}'][f'mrr_{key_src}'] = our_method[key_method]['MRR'][key_src]
            if key_src == 'attn_based':
                combined_metrics[f'ours_{key_method}']['mrr_correct'] = our_method[key_method]['MRR'][key_src+'_correct']
                combined_metrics[f'ours_{key_method}']['mrr_wrong'] = our_method[key_method]['MRR'][key_src+'_wrong']
                combined_metrics['GPT']['mrr_correct'] =  baseline_gpt['mrr_correct'] if 'mrr_correct' in baseline_gpt else baseline_gpt['mrr']
                combined_metrics['GPT']['mrr_wrong'] =  baseline_gpt['mrr_wrong'] if 'mrr_wrong' in baseline_gpt else baseline_gpt['mrr']
                combined_metrics['rules']['mrr_correct'] =  baseline_rules['mrr_correct'] if 'mrr_correct' in baseline_rules else baseline_rules['mrr']
                combined_metrics['rules']['mrr_wrong'] =  baseline_rules['mrr_wrong'] if 'mrr_wrong' in baseline_rules else baseline_rules['mrr']
            combined_metrics['GPT']['mrr_'+key_src] =  baseline_gpt['mrr']
            combined_metrics['rules']['mrr_'+key_src] =  baseline_rules['mrr']
        labels.append(' '*i)
        values.append(0)
    axs[2,0].bar(labels[:-1], values[:-1])
    axs[2,0].set_title('Explanation MRR comparison')
    
    fig.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'Seen_{:02d}'.format(num_train_actions), persona, prefix+'assistance_comparison.png'))
    
    return combined_metrics

def visualize_combined_results(combined_metrics, original_logs_dir, logs_prefix):
    method_labels = ['Rules', 'GPT', 'NO Concepts', 'TAACo', 'TAACo (Custom Concepts)', 'TAACo (Oracle)']
    methods = ['rules','GPT','ours_with_no_concepts','ours_with_gpt','ours_with_model','ours_with_user']
    methods_wo_expl = []
    accuracies = [combined_metrics[method]['accuracy'] for method in methods]
    f1s = [combined_metrics[method]['f1_macro'] for method in methods]
    subjective_metrics = list(combined_metrics['ours_with_model']['subj'][0].keys())
    subjective_metrics.remove('total')
    mrrs = {}
    for i,key_src in enumerate(['mrr_output_based', 'mrr_gt_based', 'mrr_attn_based', 'mrr_correct', 'mrr_wrong']):
        mrrs[key_src] = [[x if x is not None else 0 for x in combined_metrics[method][key_src]] for method in methods if method not in methods_wo_expl]
    
    def plot_all(errortype):
        
        def error_bars(data):
            if errortype == 'minmax':
                return [np.max(x)-np.min(x) for x in data]
            elif errortype == 'std':
                return [np.std(x) for x in data]
            else:
                return [0]*len(data)
        
        fig, axs = plt.subplots(2, 3, figsize=(25, 15))
        p = axs[0,0].bar(method_labels, [np.mean(x) for x in accuracies], yerr=error_bars(accuracies), color=method_color_list)
        axs[0,0].bar_label(p, label_type='center', fontsize=16, fmt="%.3f")
        axs[0,0].set_title('Accuracy Comparison')
        
        ignore_methods = ['ours_with_no_concepts', 'ours_with_model']
        methods_shortlist = [m for m in methods if m not in ignore_methods]
        subjectives = [combined_metrics[method]['subj'] for method in methods_shortlist]
        
        bases = [0 for _ in subjectives]
        subjective_metrics.reverse()
        for metric in subjective_metrics:
            if metric == 'correct':
                continue
            y_vals = [np.mean([x[metric]/x['total'] if x['total']>0 else 0 for x in subj]) for subj in subjectives]
            p = axs[0,1].bar(methods_shortlist, y_vals, bottom=bases, color=color_list[subjective_metrics.index(metric)], edgecolor=color_list_light[subjective_metrics.index(metric)], hatch=hatches[subjective_metrics.index(metric)], label=metric)
            bases = [b+y for b,y in zip(bases, y_vals)]
            axs[0,1].bar_label(p, label_type='center', fontsize=14, fmt="%.3f")
        axs[0,1].legend(loc='upper right', prop={'size': 16})
        axs[0,1].set_title('Error Analysis')
        
        def offset_by_method(methodname):
            num_total = len(methods_shortlist)
            idx_method = methods_shortlist.index(methodname)
            return 0.7*(idx_method/num_total)-0.35
        
        width = 0.7/len(methods_shortlist) - 0.01
        idx_metric = 0
        for metric in subjective_metrics:
            if metric == 'correct':
                continue
            y_vals = [np.mean([x[metric]/x['total'] if x['total']>0 else 0 for x in subj]) for subj in subjectives]
            y_errs = [np.std([x[metric]/x['total'] if x['total']>0 else 0 for x in subj]) for subj in subjectives]
            p = axs[0,2].bar([idx_metric+offset_by_method(m) for m in methods_shortlist], y_vals, yerr=y_errs, width=width, color=[method_color_list_lines_dict[m] for m in methods_shortlist])
            axs[0,2].bar_label(p, label_type='center', fontsize=16, fmt="%.2f")
            idx_metric += 1
        axs[0,2].set_xticks([idx for idx in range(len(subjective_metrics)-1)], labels = [s for s in subjective_metrics if s != 'correct'])
        axs[0,2].set_title('Error Analysis')
        
        method_labels_w_expl = method_labels
        for i,key_src in enumerate(['mrr_correct', 'mrr_wrong', 'mrr_attn_based']):
            p = axs[1,i].bar(method_labels_w_expl, [np.mean(x) for x in mrrs[key_src]], yerr=error_bars(mrrs[key_src]), color=method_color_list)
            axs[1,i].bar_label(p, label_type='center', fontsize=16, fmt="%.3f")
            axs[1,i].set_title(f'Explanation Accuracy ({key_src})')
            axs[1,i].set_ylim(0,1)
        fig.tight_layout()
        return fig

    print("Saving the combined plots")
    plot_all('std').savefig(os.path.join(original_logs_dir, 'std_'+logs_prefix+'assistance_comparison.png'))    
    json.dump(combined_metrics, open(os.path.join(original_logs_dir, logs_prefix+'assistance_comparison.json'), 'w'), indent=4)
    
def visualize_online(dirpath, prefix):
    step_dirs = [x for x in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, x)) and 'Seen_' in x]
    step_dirs = sorted(step_dirs, key=lambda x: int(x.split('_')[-1]))
    method_labels = [ 'GPT','Rules', 'NO Concepts', 'TAACo', 'TAACo (Custom Concepts)', 'TAACo (Oracle)']
    methods = ['GPT','rules','ours_with_no_concepts','ours_with_gpt','ours_with_model','ours_with_user']
    ignore_methods = ['ours_with_no_concepts', 'ours_with_model']
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))
    fig2, axs2 = plt.subplots(3, 3, figsize=(25, 20))
    axs = axs.flatten()
    axs2 = axs2.flatten()
    remaining_runs = []
    results = {}
    for method, method_label in zip(methods, method_labels):
        results[method] = {}
        if method in ignore_methods: continue
        train_examples = []
        accuracy_means = []
        accuracy_stds = []
        expl_acc_means = []
        expl_acc_stds = []
        expl_acc_correct_means = []
        expl_acc_correct_stds = []
        expl_acc_wrong_means = []
        expl_acc_wrong_stds = []
        error_breakdowns = {}
        error_breakdowns["unnecessary_interaction"] = {'mean':[], 'std':[]}
        error_breakdowns["delay_or_disturbance"] = {'mean':[], 'std':[]}
        error_breakdowns["task_not_done"] = {'mean':[], 'std':[]}
        error_breakdowns["performed_prohibioted_task"] = {'mean':[], 'std':[]}
        for step_dir in step_dirs:
            step_num = step_dir.split('_')[-1]
            # if step_num == '50': continue
            filepath = os.path.join(os.path.join(dirpath, step_dir), prefix+'_'+step_num+'assistance_comparison.json')
            if not os.path.exists(filepath): continue
            step_results = json.load(open(os.path.join(dirpath, step_dir, prefix+'_'+step_num+'assistance_comparison.json')))
            remaining_runs += step_results['left_to_do']
            
            if method == 'ours_with_no_concepts' and method not in step_results: continue
            train_examples.append(int(step_num))
            
            accuracy_means.append(np.mean(step_results[method]['accuracy']))
            accuracy_stds.append(np.std(step_results[method]['accuracy']))
            expl_acc_means.append(np.mean(step_results[method]['mrr_attn_based']))
            expl_acc_stds.append(np.std(step_results[method]['mrr_attn_based']))
            expl_acc_correct_means.append(np.mean(step_results[method]['mrr_correct']))
            expl_acc_correct_stds.append(np.std(step_results[method]['mrr_correct']))
            expl_acc_wrong_means.append(np.mean(step_results[method]['mrr_wrong']))
            expl_acc_wrong_stds.append(np.std(step_results[method]['mrr_wrong']))
            for error_type in error_breakdowns.keys():
                res_list = [(r[error_type]/(r["total"]+1e-8)) for r in step_results[method]['subj']]
                error_breakdowns[error_type]['mean'].append(np.mean(res_list))
                error_breakdowns[error_type]['std'].append(np.std(res_list)) 
                    
        if method == 'GPT' and train_examples[-1] == 50:
            train_examples = train_examples[:-1]
            accuracy_means = accuracy_means[:-1]
            accuracy_stds = accuracy_stds[:-1]
            expl_acc_means = expl_acc_means[:-1]
            expl_acc_stds = expl_acc_stds[:-1]
            expl_acc_correct_means = expl_acc_correct_means[:-1]
            expl_acc_correct_stds = expl_acc_correct_stds[:-1]
            expl_acc_wrong_means = expl_acc_wrong_means[:-1]
            expl_acc_wrong_stds = expl_acc_wrong_stds[:-1]
            for error_type in error_breakdowns.keys():
                error_breakdowns[error_type]['mean'] = error_breakdowns[error_type]['mean'][:-1]
                error_breakdowns[error_type]['std'] = error_breakdowns[error_type]['std'][:-1]
        results[method]['accuracy'] = {'mean':accuracy_means, 'std':accuracy_stds}
        axs[0].errorbar(train_examples, accuracy_means, yerr=accuracy_stds, label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=8, linewidth=2)
        axs[0].set_title('Accuracy')
        axs[0].legend(fontsize=20)
        results[method]['expl_accuracy'] = {'mean':expl_acc_means, 'std':expl_acc_stds}
        axs[1].errorbar(train_examples, expl_acc_means, yerr=expl_acc_stds, label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=8, linewidth=2)
        axs[1].set_title('Explanation Accuracy')
        axs[1].legend(fontsize=20)
        results[method]['expl_accuracy_correct'] = {'mean':expl_acc_correct_means, 'std':expl_acc_correct_stds}
        axs[2].errorbar(train_examples, expl_acc_correct_means, yerr=expl_acc_correct_stds, label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=8, linewidth=2)
        axs[2].set_title('Explanation Accuracy (Correct)')
        axs[2].legend(fontsize=20)
        results[method]['expl_accuracy_wrong'] = {'mean':expl_acc_wrong_means, 'std':expl_acc_wrong_stds}
        axs[3].errorbar(train_examples, expl_acc_wrong_means, yerr=expl_acc_wrong_stds, label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=8, linewidth=2)
        axs[3].set_title('Explanation Accuracy (Wrong)')
        axs[3].legend(fontsize=20)
        for i,error_type in enumerate(error_breakdowns.keys()):
            axs[i+4].errorbar(train_examples, error_breakdowns[error_type]['mean'], yerr=error_breakdowns[error_type]['std'], label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=10)
            axs[i+4].set_title(f'{error_type} Error')
            axs[i+4].legend(fontsize=20)
        if method == 'ours_with_user': continue
        for i in range(len(axs2)):
            axs2[i].errorbar(train_examples[:i+1], accuracy_means[:i+1], yerr=accuracy_stds[:i+1], label=method_label, color=method_color_list_lines[methods.index(method)], fmt=f'-{method_marker_list[methods.index(method)]}', markersize=8, capsize=8, linewidth=2)
            axs2[i].set_title('Accuracy')
            axs2[i].set_xlim(0, 45)
            axs2[i].set_ylim(0.3, 0.8)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(os.path.join(dirpath, prefix+'online_assistance_comparison.png'))
    fig2.savefig(os.path.join(dirpath, prefix+'animate_assistance_comparison.png'))
    json.dump(remaining_runs, open(os.path.join(dirpath, prefix+'left_to_do.json'), 'w'), indent=4)
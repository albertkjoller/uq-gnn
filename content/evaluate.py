
import glob, os
from PIL import Image
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from content.modules.Losses import NIGLoss
import re

# Found here: https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
def make_gif(img_dir, filename, duration=150):
    frames = [Image.open(image) for image in glob.glob(f"{img_dir}/*.png")]
    epoch_frame = [int(image.split("\\")[-1].split(".")[0]) for image in sorted(glob.glob(f"{img_dir}/*.png"))]
    frames = list(list(zip(*sorted(zip(epoch_frame, frames), key=lambda x: x[0])))[1])
    frame_one = frames[0]
    for _ in range(20):
        frames.insert(0, frame_one)
    cwd = os.getcwd()
    os.chdir(img_dir)
    frame_one.save(f"../{filename}", format="GIF", append_images=frames, save_all=True, duration=duration, loop=0)
    print(f"Created GIF: {filename}.gif")
    os.chdir(cwd)


def evidential_prediction(outputs):
    gamma, v, alpha, beta = outputs[: ,0], outputs[: ,1], outputs[: ,2], outputs[: ,3  ]# np.split(outputs, 4, axis=1)
    pred = gamma
    alea = beta /(alpha -1)
    epi = beta /( v *(alpha -1))
    return pred, alea.flatten().tolist(), epi.flatten().tolist()


def get_performance(df_summary, experiments):

    performance_dict = {}

    for exp in experiments:

        summary = df_summary[df_summary['Experiment'] == exp]

        RMSE = mean_squared_error(summary['target'], summary['prediction'], squared=False)
        exp_dict = {'RMSE': RMSE, 'NLL': {}}

        # If evidential:
        if summary['Model'][0] == 'evidential':

            evidential_params = torch.stack([torch.Tensor(summary[param]) for param in ['gamma', 'v', 'alpha', 'beta']], dim=1)

            NIG = NIGLoss(0.0) # Lambda set to 0.0 as only NIG NLL is extracted.
            exp_dict['NLL'] = float(NIG(evidential_params, torch.Tensor(summary['target']))[1]['NLL'])
            # exp_dict['NLL']['EPI_NLL'] = - torch.mean(torch.stack([Normal(loc=summary['gamma'][i], scale=summary['epistemic'][i]).log_prob(summary['target'][i]) for i in range(len(summary))]))
            # exp_dict['NLL']['ALEA_NLL'] = - torch.mean(torch.stack([Normal(loc=summary['gamma'][i], scale=summary['aleatoric'][i]).log_prob(summary['target'][i]) for i in range(len(summary))]))
            # exp_dict['NLL']['COMBINED_NLL'] = - torch.mean(torch.stack([Normal(loc=summary['gamma'][i]*2, scale=summary['epistemic'][i]+summary['aleatoric'][i]).log_prob(summary['target'][i]) for i in range(len(summary))]))

        if summary['Model'][0] == 'baseline':

            # NLL based on mu and sigma predictions
            exp_dict['NLL'] = - np.mean([scipy.stats.norm.logpdf(summary['target'][i], loc=summary['prediction'][i], scale=summary['epistemic'][i]) for i in range(len(summary))])

        performance_dict[exp] = exp_dict

    return performance_dict



def get_prediction_summary(loaders_dict, model, exp):

    target, prediction, aleatoric, epistemic, entropy, id_ood, gamma, v, alpha, beta = [], [], [], [], [], [], [], [], [], []
    # looping each dataset (ID or OOD)
    for dataset_type, loaders in loaders_dict.items():
        # dataset_type is definition whether dataset is in our out of distribution
        # TODO: CHANGE TO TEST INSTEAD OF TRAIN
        evidential_params, model_type = [], []
        num_data = 0
        for idx_batch, batch in enumerate(loaders['train']):
            num_data += len(batch.target)
            target.extend(batch.target)
            outputs = model(batch)

            if model.model_type == 'evidential':
                evidential_params.extend(outputs)
                pred, alea, epi = evidential_prediction(outputs.detach().numpy())
                prediction.extend(pred)
                aleatoric.extend(alea)
                epistemic.extend(epi)
                entropy.extend((0.5 * np.log(2 * np.pi * np.exp(1.) * (np.array(epi) ** 2))).tolist())


            elif model.model_type == 'baseline':
                # Output matching that of evidential
                evidential_params.extend(torch.zeros(outputs.shape[0], outputs.shape[1 ] *2))
                prediction.extend(outputs[: ,0].detach().numpy()) # mu
                aleatoric.extend(outputs[: ,1].detach().numpy()) # var
                epistemic.extend(outputs[: ,1].detach().numpy()) # var
                entropy.extend((0.5 * np.log(2 * np.pi * np.exp(1.) * (outputs[: ,1].detach().numpy() ** 2))).tolist())


            else: # todo: implement for ensemble etc
                raise NotImplementedError

        # finished batch, adding the last
        id_ood.extend([dataset_type ] *num_data)
        model_type.extend([model.model_type ] *num_data)
        evidential_params = torch.stack(evidential_params).detach().numpy()
        gamma_param, v_param, a_param, b_param = evidential_params[:, 0], evidential_params[:, 1], evidential_params[:, 2], evidential_params[:, 3]
        gamma.extend(gamma_param)
        v.extend(v_param)
        alpha.extend(a_param)
        beta.extend(b_param)
    # if gaussian: (only one MSE)
    # if evidential: (one for alea (based on the sigma), one for epi (based on the sigma), one combined (based on both sigmas), finally, NIG loss)

    # todo: get NLL from loss function and add
    # errors = get_errors(target, prediction, extra_params = (evidential_params, aleatoric, epistemic))
    # RMSE, NIG_nll = errors

    # i.e. same experiment and model type
    summary = {'Experiment': exp, 'Model': model.model_type,
               'target': target, 'prediction': prediction,
               'aleatoric': aleatoric, 'epistemic': epistemic,
               'gamma': prediction, 'v': v, 'alpha': alpha, 'beta': beta,
               'Entropy': entropy ,'ID or OOD': id_ood}
    # 'error': {'RMSE': RMSE}}

    # Expand error dict if evidential
    # if model.model_type == 'evidential':
    #    summary['error']['NIG_NLL'] = NIG_nll

    return summary  # , model.model_type



def error_percentile_plot(df_summary, hue_by, hue_by_list, save_path):

    # general dataframe
    df_cutoff = pd.DataFrame(columns=[hue_by, "Percentile", "RMSE"])    # from low to high conf
    percentiles = np.arange(100) / 100.
    # for exp, summary in summary_dict.items():
    for hue in hue_by_list:
        single_df_summary = df_summary[df_summary[hue_by]==hue]
        # df_summary = pd.DataFrame.from_dict(summary)
        # sorting based on uncertainty (high to low uncertainty)
        single_df_summary = single_df_summary.sort_values("epistemic", ascending=False)
        cutoff_inds = (percentiles * single_df_summary.shape[0]).astype(int)
        # the error, RMSE
        single_df_summary["Error"] = np.abs(single_df_summary["target"] - single_df_summary["prediction"])
        # take mean RMSE for cutoffs of higher uncertainty
        mean_error = [single_df_summary[cutoff:]["Error"].mean() for cutoff in cutoff_inds]
        df_single_cutoff = pd.DataFrame({hue_by: hue, 'Percentile': percentiles, 'RMSE': mean_error})
        df_cutoff = df_cutoff.append(df_single_cutoff)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Percentile", y="RMSE", hue=hue_by, data=df_cutoff.reset_index())
    plt.savefig(os.path.join(save_path, f"error_percentile.png"))
    #plt.show()
    plt.close()



def calibration_plot(df_summary, hue_by, hue_by_list, save_path):
    # general dataframe
    df_calibration = pd.DataFrame(columns=[hue_by, "Expected Conf.", "Observed Conf."])
    # from low to high conf
    expected_conf = np.arange(41) / 40.
    for hue in hue_by_list:
        single_df_summary = df_summary[df_summary[hue_by] == hue]
        observed_conf = []
        for p in expected_conf:
            # point percentage function, the z value given confidence
            # the lower tail
            lower_z = scipy.stats.norm.ppf(( 1 -p ) /2, loc=single_df_summary['prediction'], scale=single_df_summary['epistemic'])
            # the higher tail
            higher_z = scipy.stats.norm.ppf(( 1 +p ) /2, loc=single_df_summary['prediction'], scale=single_df_summary['epistemic'])
            # values within the confidence, multiplying T/F list to find union
            obs_c = np.multiply(lower_z < single_df_summary["target"], single_df_summary["target"] < higher_z).mean()
            observed_conf.append(obs_c)

        df_single = pd.DataFrame({hue_by: hue, 'Expected Conf.': expected_conf, 'Observed Conf.': observed_conf})
        df_calibration = df_calibration.append(df_single)

    # the desired
    df_truth = pd.DataFrame({hue_by: 'Ideal calibration', 'Expected Conf.': expected_conf, 'Observed Conf.': expected_conf})
    # df_calibration = df_calibration.append(df_truth)
    df_calibration.reset_index(drop=True, inplace=True)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Expected Conf.", y="Observed Conf.", data=df_truth, label='Ideal calibration', color='black', linestyle='--')
    sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue=hue_by, data=df_calibration.reset_index())
    plt.savefig(os.path.join(save_path, f"calibration.png"))
    #plt.show()
    plt.close()


def plot_results(df_summary, experiments, RMSE_NLL_COMBINED = False):

    # Getting performance dictionary (NLL and MSE)
    performance_dict = get_performance(df_summary, experiments)
    performance_df = pd.DataFrame.from_dict(performance_dict)

    # Plotting performance
    if RMSE_NLL_COMBINED:
        performance_df.plot(kind='bar', rot=0.0) # maybe split into RMSE and NLL instead

    else:
        axes = performance_df.T.plot(subplots=True, layout=(1,2), kind='bar', rot=0.0, legend=None)
        for ax in axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')

    latex = performance_df.to_latex()

    return latex


def in_ood_boxplot(df_summary, hue_by, hue_by_list, save_path):

    sns.catplot(x=hue_by, y='Entropy', hue="ID or OOD", data=df_summary, kind="box")  # , showfliers=False)
    plt.savefig(os.path.join(save_path, f"box.png"))
    #plt.show()
    plt.close()


def in_ood_hist(df_summary, hue_by, hue_by_list, save_path):

    # general dataframe
    for hue in hue_by_list:
        single_df_summary = df_summary[df_summary[hue_by] == hue]
        sns.histplot(x='Entropy', hue="ID or OOD", data=single_df_summary, kde=True)
        plt.savefig(os.path.join(save_path, f"hist_{hue}.png"))
        #plt.show()
        plt.close()



def evaluate_model(loaders_dict, models, experiments, args):
    plt.style.use('ggplot')
    df_summary = pd.DataFrame()
    # looping each experiment
    for idx, exp in enumerate(experiments):
        # looping each experiment, getting data on each dataset (ID/OOD)
        summary = get_prediction_summary(loaders_dict, models[exp], exp)
        df_single_summary = pd.DataFrame.from_dict(summary)
        df_summary = df_summary.append(df_single_summary)

    if len(set(list(df_summary['Model']))) == 1: # if 1 model type then comparing multiple experiments
        hue_by = 'Experiment' # experiment name will differentiate
        hue_by_list = list(set(list(df_summary['Experiment'])))
    else: # else comparing accross different models
        hue_by = 'Model' # model type will differentiate
        hue_by_list = list(set(list(df_summary['Model'])))

    # generating save folder
    words = []
    for exp in experiments:
        words.extend(re.split(' |_', exp))
    save_name = 'eval_' + "_".join(sorted(set(words), key=words.index))
    try:
        os.makedirs('results' + f"/{save_name}")
    except Exception as e:
        print('Save folder already exists')
    save_path = 'results' + f"/{save_name}"


    # RMSE as a function of percentile included sigma values
    #   - including sigma's from all, to only highest sigma's (based on %)
    #   - desire constant/inverse trend, no fluctuations between sigma and error
    error_percentile_plot(df_summary, hue_by, hue_by_list, save_path)

    # % correct predictions as a function of increasing confidence interval
    #   - we want a linear trend, so estimated confidence matches expected
    calibration_plot(df_summary, hue_by, hue_by_list, save_path)

    # id multiple datasets of ID and OOD
    if len(set(args.id_ood)) != 1:
        # entropy of in and out of distribution boxplot
        #   - a difference in entropy between in and out is desired
        in_ood_boxplot(df_summary, hue_by, hue_by_list, save_path)

        # histogram of entropy from in and out of distribution data
        in_ood_hist(df_summary, hue_by, hue_by_list, save_path)


    # histogram of entropy from in and out of distribution data
    # in_ood_hist(df_summary, hue_by, hue_by_list)

    latex = plot_results(df_summary, experiments)


    # RMSE, NIG_nll = errors
    # Expand error dict if evidential
    # if model.model_type == 'evidential':
    #    summary['error']['NIG_NLL'] = NIG_nll

    # Aleatoric NLL -> likelihood tarrget, based on Normal(gamma, aleatoric)
    # Epistemic NLL -> likelihood tarrget, based on Normal(gamma, Epistemic)
    # Combined --> target Normal(gamma, aleatoric, epistemic)

    # PHIL: Make table func



'''
if __name__ == '__main__':

    experiment_name = 'BASELINE_debug/baseline'

    # when running with args
    make_gif(f"../results/{experiment_name}/BASELINE", 'baseline.gif', duration=100)
    #make_gif(f"../results/{experiment_name}/EPISTEMIC", 'epistemic.gif', duration=100)
    make_gif(f"../results/{experiment_name}/PARAMS", 'parameters.gif', duration=100)





def error_conf_plot(summary, hue_by):

    # general dataframe
    df_cutoff = pd.DataFrame(columns=[hue_by, "Expected Conf.", "RMSE"])    # from low to high conf
    df_summary = pd.DataFrame.from_dict(summary)
    # the error, RMSE
    df_summary["Error"] = np.abs(df_summary["target"] - df_summary["prediction"])
    # take mean RMSE for cutoffs of higher uncertainty

    #mean_error = [df_summary[cutoff:]["Error"].mean() for cutoff in cutoff_inds]
    expected_conf = np.arange(41) / 40.
    observed_rmse = []
    # go inverse of expected, start 0 (full z range then decrease it)
    for p in expected_conf:
        # point percentage function, the z value given confidence
        # the lower tail
        lower_z = scipy.stats.norm.ppf((1-p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # the higher tail
        higher_z = scipy.stats.norm.ppf((1+p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # values within the confidence, multiplying T/F list to find union
        obs = np.multiply(lower_z < summary["target"], summary["target"] < higher_z)
        # sum rmse
        sum_rmse = df_summary["Error"][np.multiply(lower_z < summary["target"], summary["target"] < higher_z)].sum()
        observed_rmse.append(sum_rmse)

    df_single_cutoff = pd.DataFrame({hue_by: summary[hue_by], 'Expected Conf.': expected_conf, 'RMSE': observed_rmse})
    df_cutoff = df_cutoff.append(df_single_cutoff)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Expected Conf.", y="RMSE", hue=hue_by, data=df_cutoff.reset_index())
    # todo save plot

'''












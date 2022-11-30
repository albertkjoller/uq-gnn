
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
    frame_one.save(f"../{filename}", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    print(f"Created GIF: {filename}.gif")
    os.chdir(cwd)


def evidential_prediction(outputs):
    gamma, v, alpha, beta = outputs[:,0], outputs[:,1], outputs[:,2], outputs[:,3]# np.split(outputs, 4, axis=1)
    pred = gamma
    alea = beta/(alpha-1)
    epi = beta/(v*(alpha-1))
    return pred.flatten().tolist(), alea.flatten().tolist(), epi.flatten().tolist()


def get_errors(target, predicted, extra_params = None):
    # using the sklearn loss functions 
    # PHIL: MSE, NLL (with and without aleatoric (lambda))
    # NLL = log_loss(target, predicted)

    NIG_NLL, epi_NLL, alea_NLL = None

    if extra_params:
        evidential_params, aleatoric, epistemic = extra_params
        NIG = NIGLoss(0.0) # Lambda set to 0.0 as only NIG NLL is extracted. 
        NIG_NLL = NIG(torch.stack(evidential_params), torch.stack(target))[1]['NLL']

    RMSE = mean_squared_error(target, predicted, squared=False)
    return (RMSE, NIG_NLL)



def get_prediction_summary(loader, model, exp):

    #batch_loss, batch_xtra_losses = [], defaultdict(list)

    target, prediction, aleatoric, epistemic, evidential_params = [], [], [], [], []
    for idx_batch, batch in enumerate(loader):
        target.extend(batch.target)
        outputs = model(batch)

        if model.model_type == 'evidential':
            evidential_params.extend(outputs)
            pred, alea, epi = evidential_prediction(outputs.detach().numpy())
            prediction.extend(pred)
            aleatoric.extend(alea)
            epistemic.extend(epi)
        
        else:
            # todo: implement for gaussian, ensemble etc
            raise NotImplementedError

    # if gaussian: (only one MSE)
    # if evidential: (one for alea (based on the sigma), one for epi (based on the sigma), one combined (based on both sigmas), finally, NIG loss)

    # todo: get NLL from loss function and add
    errors = get_errors(target, prediction, extra_params = (evidential_params, aleatoric, epistemic))
    RMSE, NIG_nll = errors


    summary = {'Experiment': exp, 'Model': model.model_type,
               'target': target, 'prediction': prediction,
               'aleatoric': aleatoric, 'epistemic': epistemic,
               'error': {'RMSE': RMSE}}

    # Expand error dict if evidential
    if model.model_type == 'evidential':
        summary['error']['NIG_NLL'] = NIG_nll
    
    return summary, model.model_type



def error_percentile_plot(summary_dict, hue_by):

    # general dataframe
    df_cutoff = pd.DataFrame(columns=[hue_by, "Percentile", "RMSE"])    # from low to high conf
    percentiles = np.arange(100) / 100.
    # todo extend to multiple models
    for exp, summary in summary_dict.items():
        df_summary = pd.DataFrame.from_dict(summary)
        # sorting based on uncertainty (high to low uncertainty)
        df_summary = df_summary.sort_values("epistemic", ascending=False)
        cutoff_inds = (percentiles * df_summary.shape[0]).astype(int)
        # the error, RMSE
        df_summary["Error"] = np.abs(df_summary["target"] - df_summary["prediction"])
        # take mean RMSE for cutoffs of higher uncertainty
        mean_error = [df_summary[cutoff:]["Error"].mean() for cutoff in cutoff_inds]
        df_single_cutoff = pd.DataFrame({hue_by: summary[hue_by], 'Percentile': percentiles, 'RMSE': mean_error})
        df_cutoff = df_cutoff.append(df_single_cutoff)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Percentile", y="RMSE", hue=hue_by, data=df_cutoff.reset_index())
    # todo save plot



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




def calibration_plot(summary, hue_by):
    # general dataframe
    df_calibration = pd.DataFrame(columns=[hue_by, "Expected Conf.", "Observed Conf."])
    # from low to high conf
    expected_conf = np.arange(41) / 40.
    observed_conf = []
    # go inverse of expected, start 0 (full z range then decrease it)
    for p in expected_conf:
        # point percentage function, the z value given confidence
        # the lower tail
        lower_z = scipy.stats.norm.ppf((1-p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # the higher tail
        higher_z = scipy.stats.norm.ppf((1+p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # values within the confidence, multiplying T/F list to find union
        obs_c = np.multiply(lower_z < summary["target"], summary["target"] < higher_z).mean()
        observed_conf.append(obs_c)

    df_single = pd.DataFrame({hue_by: summary[hue_by], 'Expected Conf.': expected_conf, 'Observed Conf.': observed_conf})
    df_calibration = df_calibration.append(df_single)

    # the desired
    df_truth = pd.DataFrame({hue_by: 'Ideal calibration', 'Expected Conf.': expected_conf, 'Observed Conf.': expected_conf})
    #df_calibration = df_calibration.append(df_truth)
    df_calibration.reset_index(drop=True, inplace=True)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Expected Conf.", y="Observed Conf.", data=df_truth, label='Ideal calibration', color='black', linestyle='--')
    sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue=hue_by, data=df_calibration.reset_index())

    # todo save plot

def get_results(summary):

    return None


def in_odd_boxplot(summary):
    a=0


def evaluate_model(loader, models, experiments):

    # todo extend to multiple models
    summary_dict = {}
    model_names = []
    for idx, exp in enumerate(experiments):
        summary, model_name = get_prediction_summary(loader, models[exp], exp)
        model_names.append(model_name)
        summary_dict[exp] = summary

    if len(set(model_names)) == 1: # if 1 model type then comparing multiple experiments
        hue_by = 'Experiment' # experiment name will differentiate
    else: # else comparing accross different models
        hue_by = 'Model' # model type will differentiate


    # RMSE as a function of percentile included sigma values
    #   - including sigma's from all, to only highest sigma's (based on %)
    #   - desire constant/inverse trend, no fluctuations between sigma and error
    error_percentile_plot(summary_dict, hue_by)

    # % correct predictions as a function of increasing confidence interval
    #   - we want a linear trend, so estimated confidence matches expected
    calibration_plot(summary_dict, hue_by)

    # entropy of in and out of distribution boxplot
    #   - a difference in entropy between in and out is desired
    in_odd_boxplot(summary_dict, hue_by)


    # RMSE as a function of increasing confidence interval, ignore, don't
    #error_conf_plot(summary)
    #todo: if modeltype uniqu is same, hue_by is experiment name, else model_yupe

    get_results(summary)

    # PHIL: Make table func




if __name__ == '__main__':

    experiment_name = 'BASELINE_debug/baseline'

    # when running with args
    make_gif(f"../results/{experiment_name}/BASELINE", 'baseline.gif', duration=100)
    #make_gif(f"../results/{experiment_name}/EPISTEMIC", 'epistemic.gif', duration=100)
    make_gif(f"../results/{experiment_name}/PARAMS", 'parameters.gif', duration=100)

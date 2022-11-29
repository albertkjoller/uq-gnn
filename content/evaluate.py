
import glob, os
from PIL import Image
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    gamma, v, alpha, beta = np.split(outputs, 4, axis=1)
    pred = gamma
    alea = beta/(alpha-1)
    epi = beta/(v*(alpha-1))
    return pred.flatten().tolist(), alea.flatten().tolist(), epi.flatten().tolist()


def get_errors(target, predicted):
    # using the sklearn loss functions
    #NLL = log_loss(target, predicted)
    RMSE = mean_squared_error(target, predicted, squared=False)
    return (RMSE)



def get_prediction_summary(loader, model, exp):

    #batch_loss, batch_xtra_losses = [], defaultdict(list)

    target, prediction, aleatoric, epistemic = [], [], [], []
    for idx_batch, batch in enumerate(loader):
        target.extend(batch.target.flatten().numpy().tolist())
        # get predicted
        outputs = model(batch).detach().numpy()
        if model.model_type == 'evidential':
            pred, alea, epi = evidential_prediction(outputs)
            prediction.extend(pred)
            aleatoric.extend(alea)
            epistemic.extend(epi)
        else:
            # todo: implement for gaussian, ensemble etc
            raise NotImplementedError


    # todo: get NLL from loss function and add
    errors = get_errors(target, prediction)
    RMSE = errors

    summary = {'model_name': exp, 'model_type': model.model_type,
                'target': target, 'prediction': prediction,
                'aleatoric': aleatoric, 'epistemic': epistemic}
                #'error': {'RMSE': RMSE, 'NLL': 0}}
    return summary


def error_conf_plot(summary):

    # general dataframe
    df_cutoff = pd.DataFrame(columns=["Model", "Percentile", "Error"])    # from low to high conf
    percentiles = np.arange(100) / 100.
    # todo extend to multiple models
    df_summary = pd.DataFrame.from_dict(summary)
    # sorting based on uncertainty (high to low uncertainty)
    df_summary = df_summary.sort_values("epistemic", ascending=False)
    cutoff_inds = (percentiles * df_summary.shape[0]).astype(int)
    # the error
    df_summary["Error"] = np.abs(df_summary["target"] - df_summary["prediction"])
    # take mean RMSE for cutoffs of higher uncertainty
    mean_error = [df_summary[cutoff:]["Error"].mean() for cutoff in cutoff_inds]
    df_single_cutoff = pd.DataFrame({'Model': summary['model_name'], 'Percentile': percentiles, 'RMSE': mean_error})
    df_cutoff = df_cutoff.append(df_single_cutoff)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Percentile", y="RMSE", hue="Model", data=df_cutoff)
    # todo save plot


def calibration_plot(summary):
    # general dataframe
    df_calibration = pd.DataFrame(columns=["Model", "Expected Conf.", "Observed Conf."])
    # from low to high conf
    expected_conf = np.arange(41) / 40.
    observed_conf = []
    # todo extend to multiple models
    # go inverse of expected, start 0 (full z range then decrease it)
    for p in expected_conf:
        # point percentage function, the z value given confidence
        # the lower tail
        lower_z = scipy.stats.norm.ppf((1-p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # the higher tail
        higher_z = scipy.stats.norm.ppf((1+p)/2, loc=summary['prediction'], scale=summary['epistemic'])
        # values within the confidence
        obs_c = np.multiply(lower_z < summary["target"], summary["target"] < higher_z).mean()
        observed_conf.append(obs_c)

    df_single = pd.DataFrame({'Model': summary['model_name'], 'Expected Conf.': expected_conf, 'Observed Conf.': observed_conf})
    df_calibration = df_calibration.append(df_single)

    # the desired
    df_truth = pd.DataFrame({'Model': 'Ideal calibration', 'Expected Conf.': expected_conf, 'Observed Conf.': expected_conf})
    df_calibration = df_calibration.append(df_truth)
    df_calibration.reset_index(drop=True, inplace=True)

    # made for plotitng multiple models confidence
    sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue="Model", data=df_calibration)
    # todo save plot


def evaluate_model(loader, model, exp):

    summary = get_prediction_summary(loader, model, exp)

    error_conf_plot(summary)

    calibration_plot(summary)




if __name__ == '__main__':

    experiment_name = 'BASELINE_debug/baseline'

    # when running with args
    make_gif(f"../results/{experiment_name}/BASELINE", 'baseline.gif', duration=100)
    #make_gif(f"../results/{experiment_name}/EPISTEMIC", 'epistemic.gif', duration=100)
    make_gif(f"../results/{experiment_name}/PARAMS", 'parameters.gif', duration=100)
